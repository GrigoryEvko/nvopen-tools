// Function: sub_1E431B0
// Address: 0x1e431b0
//
__int64 __fastcall sub_1E431B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r13
  unsigned __int64 v5; // r9
  __int64 *v6; // rcx
  int v7; // eax
  char v9; // dl
  unsigned int v10; // esi
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 *v13; // rsi
  __int64 *v14; // rdx
  __int64 v15; // [rsp+0h] [rbp-90h] BYREF
  __int64 *v16; // [rsp+8h] [rbp-88h]
  _BYTE *v17; // [rsp+10h] [rbp-80h]
  __int64 v18; // [rsp+18h] [rbp-78h]
  int v19; // [rsp+20h] [rbp-70h]
  _BYTE v20[104]; // [rsp+28h] [rbp-68h] BYREF

  v3 = *(_QWORD *)(a1 + 40);
  v17 = v20;
  v15 = 0;
  v16 = (__int64 *)v20;
  v18 = 8;
  v19 = 0;
  v4 = sub_1E69D00(v3, a2);
LABEL_2:
  v5 = (unsigned __int64)v17;
  v6 = v16;
  v7 = **(unsigned __int16 **)(v4 + 16);
  if ( v7 == 45 )
    goto LABEL_8;
LABEL_3:
  if ( v7 )
  {
LABEL_4:
    if ( (__int64 *)v5 != v6 )
      _libc_free(v5);
    return v4;
  }
  while ( 1 )
  {
LABEL_8:
    if ( (__int64 *)v5 != v6 )
      goto LABEL_9;
    v13 = &v6[HIDWORD(v18)];
    if ( v13 != v6 )
      break;
LABEL_23:
    if ( HIDWORD(v18) < (unsigned int)v18 )
    {
      ++HIDWORD(v18);
      *v13 = v4;
      v6 = v16;
      ++v15;
      v5 = (unsigned __int64)v17;
      goto LABEL_10;
    }
LABEL_9:
    sub_16CCBA0((__int64)&v15, v4);
    v5 = (unsigned __int64)v17;
    v6 = v16;
    if ( !v9 )
      goto LABEL_4;
LABEL_10:
    v10 = *(_DWORD *)(v4 + 40);
    if ( v10 > 1 )
    {
      v11 = *(_QWORD *)(v4 + 32);
      v12 = 1;
      while ( *(_QWORD *)(a1 + 920) != *(_QWORD *)(v11 + 40LL * (unsigned int)(v12 + 1) + 24) )
      {
        v12 = (unsigned int)(v12 + 2);
        if ( v10 <= (unsigned int)v12 )
          goto LABEL_7;
      }
      v4 = sub_1E69D00(*(_QWORD *)(a1 + 40), *(unsigned int *)(v11 + 40 * v12 + 8));
      goto LABEL_2;
    }
LABEL_7:
    v7 = **(unsigned __int16 **)(v4 + 16);
    if ( v7 != 45 )
      goto LABEL_3;
  }
  v14 = 0;
  while ( v4 != *v6 )
  {
    if ( *v6 == -2 )
      v14 = v6;
    if ( v13 == ++v6 )
    {
      if ( !v14 )
        goto LABEL_23;
      *v14 = v4;
      v5 = (unsigned __int64)v17;
      --v19;
      v6 = v16;
      ++v15;
      goto LABEL_10;
    }
  }
  return v4;
}
