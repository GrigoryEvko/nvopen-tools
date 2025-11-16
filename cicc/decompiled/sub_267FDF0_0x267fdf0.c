// Function: sub_267FDF0
// Address: 0x267fdf0
//
__int64 __fastcall sub_267FDF0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 v6; // r15
  __int64 v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // r12
  _BYTE *v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rdi
  unsigned int v14; // ecx
  __int64 *v15; // rsi
  __int64 v16; // r8
  int v17; // esi
  int v18; // r10d

  result = *(_QWORD *)(a2 + 120);
  v3 = *(_QWORD *)(result + 16);
  if ( v3 )
  {
    while ( 1 )
    {
      v11 = *(_BYTE **)(v3 + 24);
      if ( *v11 <= 0x1Cu )
        break;
      v6 = *a1;
      if ( !*a1 || !*(_DWORD *)(v6 + 40) )
        goto LABEL_5;
      result = sub_B43CB0(*(_QWORD *)(v3 + 24));
      v12 = *(unsigned int *)(v6 + 24);
      v13 = *(_QWORD *)(v6 + 8);
      if ( (_DWORD)v12 )
      {
        v14 = (v12 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v15 = (__int64 *)(v13 + 8LL * v14);
        v16 = *v15;
        if ( result != *v15 )
        {
          v17 = 1;
          while ( v16 != -4096 )
          {
            v18 = v17 + 1;
            v14 = (v12 - 1) & (v17 + v14);
            v15 = (__int64 *)(v13 + 8LL * v14);
            v16 = *v15;
            if ( result == *v15 )
              goto LABEL_14;
            v17 = v18;
          }
          goto LABEL_9;
        }
LABEL_14:
        result = v13 + 8 * v12;
        if ( v15 != (__int64 *)result )
        {
LABEL_5:
          v7 = sub_B43CB0((__int64)v11);
LABEL_6:
          v10 = sub_267FA80(a2, v7);
          result = *((unsigned int *)v10 + 2);
          if ( result + 1 > (unsigned __int64)*((unsigned int *)v10 + 3) )
          {
            sub_C8D5F0((__int64)v10, v10 + 2, result + 1, 8u, v8, v9);
            result = *((unsigned int *)v10 + 2);
          }
          *(_QWORD *)(*v10 + 8 * result) = v3;
          ++*((_DWORD *)v10 + 2);
          goto LABEL_9;
        }
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          return result;
      }
      else
      {
LABEL_9:
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          return result;
      }
    }
    v7 = 0;
    goto LABEL_6;
  }
  return result;
}
