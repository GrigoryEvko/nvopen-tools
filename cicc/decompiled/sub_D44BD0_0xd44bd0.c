// Function: sub_D44BD0
// Address: 0xd44bd0
//
unsigned __int64 __fastcall sub_D44BD0(__int64 *a1, unsigned __int64 a2)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // r12
  unsigned __int8 v5; // al
  __int64 v6; // rdx
  __int64 *v7; // rdi
  unsigned __int8 v8; // al
  __int64 *v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 v17; // [rsp+10h] [rbp-80h]
  unsigned __int64 v18; // [rsp+18h] [rbp-78h]
  __int64 v19; // [rsp+28h] [rbp-68h] BYREF
  unsigned __int64 v20[12]; // [rsp+30h] [rbp-60h] BYREF

  v2 = *a1;
  v3 = a1[6];
  v4 = a1[7];
  v17 = a1[4];
  v18 = a1[5];
  v19 = a1[1];
  if ( v3 )
  {
    v5 = *(_BYTE *)(v3 - 16);
    if ( (v5 & 2) != 0 )
    {
      v7 = *(__int64 **)(v3 - 32);
      v6 = *(unsigned int *)(v3 - 24);
    }
    else
    {
      v6 = (*(_WORD *)(v3 - 16) >> 6) & 0xF;
      v7 = (__int64 *)(v3 - 8LL * ((v5 >> 2) & 0xF) - 16);
    }
    if ( &v7[v6] == sub_D338A0(v7, (__int64)&v7[v6], v2) )
    {
      if ( v4 )
        goto LABEL_6;
LABEL_13:
      v4 = 0;
      goto LABEL_9;
    }
  }
  v3 = 0;
  if ( !v4 )
    goto LABEL_13;
LABEL_6:
  v8 = *(_BYTE *)(v4 - 16);
  if ( (v8 & 2) != 0 )
  {
    v9 = *(__int64 **)(v4 - 32);
    v10 = *(unsigned int *)(v4 - 24);
  }
  else
  {
    v10 = (*(_WORD *)(v4 - 16) >> 6) & 0xF;
    v9 = (__int64 *)(v4 - 8LL * ((v8 >> 2) & 0xF) - 16);
  }
  if ( &v9[v10] != sub_D338A0(v9, (__int64)&v9[v10], v2) )
    goto LABEL_13;
LABEL_9:
  v20[5] = v4;
  v20[0] = a2;
  v20[2] = v17;
  v20[4] = v3;
  v20[3] = v18;
  v20[1] = -1;
  sub_FD9690(v2 + 976, v20);
  v20[0] = a2 | 4;
  v11 = sub_D40250(v2, v20);
  return sub_D44380(v11, &v19, v12, v13, v14, v15);
}
