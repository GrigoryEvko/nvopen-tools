// Function: sub_24159D0
// Address: 0x24159d0
//
__int64 __fastcall sub_24159D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  unsigned int v4; // r8d
  __int64 v5; // rdi
  int v6; // r14d
  __int64 *v7; // r12
  unsigned int v8; // edx
  _QWORD *v9; // rax
  __int64 v10; // r10
  char **v11; // r12
  __int64 result; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 **v15; // rax
  int v16; // eax
  int v17; // edx
  int v18; // esi
  __int64 v19; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v20; // [rsp+18h] [rbp-28h] BYREF

  v2 = a2;
  v19 = a2;
  if ( (*(_BYTE *)a2 > 0x1Cu || *(_BYTE *)a2 == 22) && !*(_BYTE *)(a1 + 145) )
  {
    v4 = *(_DWORD *)(a1 + 200);
    if ( v4 )
    {
      v5 = *(_QWORD *)(a1 + 184);
      v6 = 1;
      v7 = 0;
      v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (_QWORD *)(v5 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
      {
LABEL_6:
        v11 = (char **)(v9 + 1);
        result = v9[1];
        if ( result )
          return result;
        goto LABEL_24;
      }
      while ( v10 != -4096 )
      {
        if ( !v7 && v10 == -8192 )
          v7 = v9;
        v8 = (v4 - 1) & (v6 + v8);
        v9 = (_QWORD *)(v5 + 16LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_6;
        ++v6;
      }
      if ( !v7 )
        v7 = v9;
      v16 = *(_DWORD *)(a1 + 192);
      ++*(_QWORD *)(a1 + 176);
      v17 = v16 + 1;
      v20 = v7;
      if ( 4 * (v16 + 1) < 3 * v4 )
      {
        if ( v4 - *(_DWORD *)(a1 + 196) - v17 > v4 >> 3 )
        {
LABEL_21:
          *(_DWORD *)(a1 + 192) = v17;
          if ( *v7 != -4096 )
            --*(_DWORD *)(a1 + 196);
          *v7 = v2;
          a2 = v19;
          v11 = (char **)(v7 + 1);
          *v11 = 0;
LABEL_24:
          if ( *(_BYTE *)a2 == 22 )
          {
            if ( *(_BYTE *)(a1 + 144) )
            {
              return sub_240F1A0(*(_QWORD *)a1, *(_QWORD *)(a2 + 8));
            }
            else
            {
              *v11 = (char *)sub_2411F50((__int64 *)a1, a2);
              sub_24141E0(a1 + 360, v11);
              return (__int64)*v11;
            }
          }
          else
          {
            result = sub_240F1A0(*(_QWORD *)a1, *(_QWORD *)(a2 + 8));
            *v11 = (char *)result;
          }
          return result;
        }
        v18 = v4;
LABEL_31:
        sub_FAA400(a1 + 176, v18);
        sub_F9D990(a1 + 176, &v19, &v20);
        v2 = v19;
        v7 = v20;
        v17 = *(_DWORD *)(a1 + 192) + 1;
        goto LABEL_21;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 176);
      v20 = 0;
    }
    v18 = 2 * v4;
    goto LABEL_31;
  }
  v13 = *(_QWORD *)(a2 + 8);
  v14 = *(_QWORD *)a1;
  if ( (unsigned __int8)(*(_BYTE *)(v13 + 8) - 15) > 1u )
    return *(_QWORD *)(v14 + 72);
  v15 = (__int64 **)sub_240F000(v14, v13);
  return sub_AC9350(v15);
}
