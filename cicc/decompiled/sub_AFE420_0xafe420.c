// Function: sub_AFE420
// Address: 0xafe420
//
__int64 __fastcall sub_AFE420(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r13d
  __int64 v6; // rax
  __int64 v7; // r14
  unsigned __int8 v9; // dl
  __int64 v10; // rcx
  __int64 v11; // rsi
  unsigned __int8 v12; // dl
  __int64 v13; // rcx
  int v14; // r13d
  int v15; // eax
  int v16; // r8d
  _QWORD *v17; // rdi
  unsigned int v18; // eax
  _QWORD *v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // [rsp+0h] [rbp-40h] BYREF
  __int64 v22; // [rsp+8h] [rbp-38h] BYREF
  char v23; // [rsp+10h] [rbp-30h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v9 = *(_BYTE *)(*a2 - 16);
    v10 = *a2 - 16;
    if ( (v9 & 2) != 0 )
      v11 = *(_QWORD *)(v6 - 32);
    else
      v11 = v10 - 8LL * ((v9 >> 2) & 0xF);
    v21 = *(_QWORD *)(v11 + 8);
    v12 = *(_BYTE *)(v6 - 16);
    if ( (v12 & 2) != 0 )
      v13 = *(_QWORD *)(v6 - 32);
    else
      v13 = v10 - 8LL * ((v12 >> 2) & 0xF);
    v14 = v4 - 1;
    v22 = *(_QWORD *)(v13 + 16);
    v23 = *(_BYTE *)(v6 + 1) >> 7;
    v15 = sub_AFB5F0(&v21, &v22);
    v16 = 1;
    v17 = 0;
    v18 = v14 & v15;
    v19 = (_QWORD *)(v7 + 8LL * v18);
    v20 = *v19;
    if ( *v19 == *a2 )
    {
LABEL_16:
      *a3 = v19;
      return 1;
    }
    else
    {
      while ( v20 != -4096 )
      {
        if ( v20 != -8192 || v17 )
          v19 = v17;
        v18 = v14 & (v16 + v18);
        v20 = *(_QWORD *)(v7 + 8LL * v18);
        if ( v20 == *a2 )
        {
          v19 = (_QWORD *)(v7 + 8LL * v18);
          goto LABEL_16;
        }
        ++v16;
        v17 = v19;
        v19 = (_QWORD *)(v7 + 8LL * v18);
      }
      if ( !v17 )
        v17 = v19;
      *a3 = v17;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
