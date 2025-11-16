// Function: sub_AE6600
// Address: 0xae6600
//
__int64 __fastcall sub_AE6600(_QWORD *a1, _BYTE *a2)
{
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // rcx
  _QWORD *v9; // rdx
  __int64 v10; // r15
  _BYTE *v11; // rsi
  _QWORD *v12; // r12
  _BYTE *v13; // r13
  _BYTE *v14; // rbx
  __int64 v15; // rdi
  _QWORD *v16; // rax
  __int64 v17; // rcx
  _QWORD *v18; // rdx
  __int64 v19; // r15
  __int64 v20; // rax
  char v21; // dl
  char v22; // dl
  _QWORD *v23; // [rsp+0h] [rbp-70h] BYREF
  int v24; // [rsp+8h] [rbp-68h]
  _BYTE v25[96]; // [rsp+10h] [rbp-60h] BYREF

  result = sub_B9F8A0(*a1);
  if ( result )
  {
    v5 = *(_QWORD *)(result + 16);
    if ( v5 )
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(v5 + 24);
        if ( *(_BYTE *)v6 != 85 )
          goto LABEL_4;
        result = *(_QWORD *)(v6 - 32);
        if ( !result )
          goto LABEL_4;
        if ( *(_BYTE *)result )
          goto LABEL_4;
        if ( *(_QWORD *)(result + 24) != *(_QWORD *)(v6 + 80) )
          goto LABEL_4;
        if ( (*(_BYTE *)(result + 33) & 0x20) == 0 )
          goto LABEL_4;
        result = *(unsigned int *)(result + 36);
        if ( (_DWORD)result != 71 && (_DWORD)result != 68 )
          goto LABEL_4;
        v7 = a1[1];
        if ( !*(_BYTE *)(v7 + 28) )
          goto LABEL_40;
        result = *(_QWORD *)(v7 + 8);
        v8 = *(unsigned int *)(v7 + 20);
        v9 = (_QWORD *)(result + 8 * v8);
        if ( (_QWORD *)result != v9 )
        {
          while ( v6 != *(_QWORD *)result )
          {
            result += 8;
            if ( v9 == (_QWORD *)result )
              goto LABEL_16;
          }
          goto LABEL_4;
        }
LABEL_16:
        if ( (unsigned int)v8 < *(_DWORD *)(v7 + 16) )
        {
          *(_DWORD *)(v7 + 20) = v8 + 1;
          *v9 = v6;
          ++*(_QWORD *)v7;
LABEL_18:
          v10 = a1[3];
          result = *(unsigned int *)(v10 + 8);
          if ( result + 1 > (unsigned __int64)*(unsigned int *)(v10 + 12) )
          {
            sub_C8D5F0(a1[3], v10 + 16, result + 1, 8);
            result = *(unsigned int *)(v10 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v10 + 8 * result) = v6;
          ++*(_DWORD *)(v10 + 8);
          v5 = *(_QWORD *)(v5 + 8);
          if ( !v5 )
            break;
        }
        else
        {
LABEL_40:
          result = sub_C8CC70(v7, *(_QWORD *)(v5 + 24));
          if ( v21 )
            goto LABEL_18;
LABEL_4:
          v5 = *(_QWORD *)(v5 + 8);
          if ( !v5 )
            break;
        }
      }
    }
  }
  if ( !a1[4] || *a2 != 2 )
    return result;
  v11 = a2 + 8;
  sub_B967C0(&v23, a2 + 8);
  v12 = v23;
  v13 = &v23[v24];
  if ( v23 == (_QWORD *)v13 )
    goto LABEL_37;
  do
  {
    while ( 1 )
    {
      v14 = (_BYTE *)*v12;
      if ( (unsigned __int8)(*(_BYTE *)(*v12 + 64LL) - 1) <= 1u )
      {
        v15 = a1[2];
        if ( *(_BYTE *)(v15 + 28) )
        {
          v16 = *(_QWORD **)(v15 + 8);
          v17 = *(unsigned int *)(v15 + 20);
          v18 = &v16[v17];
          if ( v16 != v18 )
          {
            while ( v14 != (_BYTE *)*v16 )
            {
              if ( v18 == ++v16 )
                goto LABEL_31;
            }
            goto LABEL_25;
          }
LABEL_31:
          if ( (unsigned int)v17 < *(_DWORD *)(v15 + 16) )
          {
            *(_DWORD *)(v15 + 20) = v17 + 1;
            *v18 = v14;
            ++*(_QWORD *)v15;
            break;
          }
        }
        v11 = (_BYTE *)*v12;
        sub_C8CC70(v15, *v12);
        if ( v22 )
          break;
      }
LABEL_25:
      if ( v13 == (_BYTE *)++v12 )
        goto LABEL_36;
    }
    v19 = a1[4];
    v20 = *(unsigned int *)(v19 + 8);
    if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v19 + 12) )
    {
      v11 = (_BYTE *)(v19 + 16);
      sub_C8D5F0(a1[4], v19 + 16, v20 + 1, 8);
      v20 = *(unsigned int *)(v19 + 8);
    }
    ++v12;
    *(_QWORD *)(*(_QWORD *)v19 + 8 * v20) = v14;
    ++*(_DWORD *)(v19 + 8);
  }
  while ( v13 != (_BYTE *)v12 );
LABEL_36:
  v13 = v23;
LABEL_37:
  result = (__int64)v25;
  if ( v13 != v25 )
    return _libc_free(v13, v11);
  return result;
}
