// Function: sub_FCDCB0
// Address: 0xfcdcb0
//
__int64 __fastcall sub_FCDCB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  int v7; // r12d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  _QWORD *v11; // r12
  char v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdx
  char v18; // cl
  unsigned __int64 v19; // r9
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdx
  _QWORD *v23; // rax
  unsigned int v24; // r9d
  int v25; // edx
  __int64 v26; // [rsp+8h] [rbp-98h]
  int v27; // [rsp+1Ch] [rbp-84h]
  __int64 v28; // [rsp+28h] [rbp-78h]
  _QWORD *v29; // [rsp+30h] [rbp-70h] BYREF
  __int64 v30; // [rsp+38h] [rbp-68h]
  _QWORD v31[2]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v32; // [rsp+50h] [rbp-50h] BYREF
  __int64 v33; // [rsp+58h] [rbp-48h]
  _QWORD v34[8]; // [rsp+60h] [rbp-40h] BYREF

  if ( *(_BYTE *)a2 != 85 )
    goto LABEL_4;
  v2 = *(_QWORD *)(a2 - 32);
  if ( !*(_BYTE *)v2 )
  {
    if ( *(_QWORD *)(v2 + 24) == *(_QWORD *)(a2 + 80)
      && (*(_BYTE *)(v2 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v2 + 36) - 9431) <= 1 )
    {
      v22 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      v23 = *(_QWORD **)(v22 + 24);
      if ( *(_DWORD *)(v22 + 32) > 0x40u )
        v23 = (_QWORD *)*v23;
      LODWORD(v32) = (_DWORD)v23;
      BYTE4(v32) = 1;
      return (__int64)v32;
    }
LABEL_4:
    BYTE4(v32) = 0;
    return (__int64)v32;
  }
  if ( *(_BYTE *)v2 != 25 )
    goto LABEL_4;
  v27 = 0;
  if ( *(char *)(a2 + 7) >= 0 )
    goto LABEL_14;
  v4 = sub_BD2BC0(a2);
  v6 = v4 + v5;
  if ( *(char *)(a2 + 7) >= 0 )
  {
    if ( (unsigned int)(v6 >> 4) )
LABEL_66:
      BUG();
LABEL_14:
    v10 = 0;
    goto LABEL_15;
  }
  if ( !(unsigned int)((v6 - sub_BD2BC0(a2)) >> 4) )
    goto LABEL_14;
  if ( *(char *)(a2 + 7) >= 0 )
    goto LABEL_66;
  v7 = *(_DWORD *)(sub_BD2BC0(a2) + 8);
  if ( *(char *)(a2 + 7) >= 0 )
    BUG();
  v8 = sub_BD2BC0(a2);
  v10 = 32LL * (unsigned int)(*(_DWORD *)(v8 + v9 - 4) - v7);
LABEL_15:
  LODWORD(v11) = 0;
  v12 = 0;
  if ( (unsigned int)((32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) - 32 - v10) >> 5) == 1 )
  {
    v21 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *(_BYTE *)v21 == 17 )
    {
      v11 = *(_QWORD **)(v21 + 24);
      if ( *(_DWORD *)(v21 + 32) > 0x40u )
        v11 = (_QWORD *)*v11;
      v12 = 1;
    }
  }
  LOBYTE(v27) = v12;
  v29 = v31;
  sub_FCD470((__int64 *)&v29, "setmaxnreg.", (__int64)"");
  v13 = sub_22416F0(v2 + 24, v29, 0, v30);
  if ( v13 != -1 )
  {
    v32 = v34;
    v26 = v13 + v30 + 3;
    sub_FCD470((__int64 *)&v32, ".sync.aligned.u32", (__int64)"");
    if ( v26 == sub_22416F0(v2 + 24, v32, v26, v33) )
    {
      v14 = *(_QWORD *)(v2 + 24);
      v15 = *(_QWORD *)(v2 + 32);
      v16 = v26 + v33;
      if ( v12 )
      {
        if ( v15 <= v16 )
        {
          if ( v16 + 1 >= v15 || *(_BYTE *)(v14 + v16) != 36 || *(_BYTE *)(v14 + v16 + 1) != 48 )
          {
LABEL_29:
            v12 = 0;
            goto LABEL_30;
          }
          v17 = v26 + v33;
        }
        else
        {
          v17 = v26 + v33;
          while ( 1 )
          {
            v18 = *(_BYTE *)(v14 + v17);
            v19 = v17++;
            if ( v18 != 32 && v18 != 9 )
              break;
            if ( v15 <= v17 )
              goto LABEL_24;
          }
          v17 = v19;
LABEL_24:
          if ( v17 + 1 >= v15 || *(_BYTE *)(v14 + v17) != 36 || *(_BYTE *)(v14 + v17 + 1) != 48 )
          {
LABEL_26:
            while ( 1 )
            {
              LOBYTE(v20) = *(_BYTE *)(v14 + v16);
              if ( (_BYTE)v20 != 32 && (_BYTE)v20 != 9 )
                break;
              if ( v15 <= ++v16 )
                goto LABEL_29;
            }
            v12 = 0;
            v24 = 0;
            if ( (unsigned int)(unsigned __int8)v20 - 48 > 9 )
              goto LABEL_30;
            while ( 1 )
            {
              v25 = (char)v20 - 48;
              if ( v24 > ~v25 / 0xAu )
                goto LABEL_29;
              ++v16;
              v24 = v25 + 10 * v24;
              if ( v15 > v16 )
              {
                v20 = *(unsigned __int8 *)(v14 + v16);
                if ( (unsigned int)(v20 - 48) <= 9 )
                  continue;
              }
              LODWORD(v11) = v24;
              v12 = 1;
              goto LABEL_30;
            }
          }
        }
        if ( v15 <= v17 + 2 || (unsigned int)*(unsigned __int8 *)(v14 + v17 + 2) - 48 > 9 )
        {
          HIDWORD(v28) = v27;
LABEL_30:
          if ( v32 != v34 )
            j_j___libc_free_0(v32, v34[0] + 1LL);
          goto LABEL_35;
        }
      }
      if ( v15 > v16 )
        goto LABEL_26;
      goto LABEL_29;
    }
    if ( v32 != v34 )
      j_j___libc_free_0(v32, v34[0] + 1LL);
  }
  v12 = 0;
LABEL_35:
  if ( v29 != v31 )
    j_j___libc_free_0(v29, v31[0] + 1LL);
  LODWORD(v28) = (_DWORD)v11;
  BYTE4(v28) = v12;
  return v28;
}
