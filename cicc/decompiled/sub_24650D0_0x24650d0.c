// Function: sub_24650D0
// Address: 0x24650d0
//
unsigned __int64 __fastcall sub_24650D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned __int8 v5; // al
  __int64 v6; // rax
  __int64 v7; // rax
  _BYTE *v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r14
  __int64 v12; // rsi
  __int64 v13; // rax
  _BYTE *v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int8 *v17; // r9
  __int64 (__fastcall *v18)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v19; // rax
  _QWORD *v21; // rax
  __int64 v22; // r9
  __int64 v23; // rdx
  int v24; // ecx
  int v25; // eax
  _QWORD *v26; // rdi
  __int64 *v27; // rax
  __int64 v28; // rax
  unsigned int *v29; // rbx
  __int64 v30; // r14
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  _BYTE *v36; // r14
  __int64 v37; // rax
  _BYTE *v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 **v42; // rax
  unsigned __int8 *v43; // [rsp+10h] [rbp-D0h]
  __int64 v44; // [rsp+10h] [rbp-D0h]
  __int64 v45; // [rsp+10h] [rbp-D0h]
  unsigned __int8 *v46; // [rsp+10h] [rbp-D0h]
  _BYTE *v47; // [rsp+20h] [rbp-C0h]
  __int64 v48; // [rsp+28h] [rbp-B8h]
  __int64 v49; // [rsp+30h] [rbp-B0h]
  int v50; // [rsp+44h] [rbp-9Ch] BYREF
  __int64 v51; // [rsp+48h] [rbp-98h]
  int v52[8]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v53; // [rsp+70h] [rbp-70h]
  _QWORD v54[4]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v55; // [rsp+A0h] [rbp-40h]

  v4 = a1;
  v48 = a2;
  v49 = *(_QWORD *)(a2 + 8);
  v5 = *(_BYTE *)(v49 + 8);
  if ( v5 == 15 )
  {
LABEL_2:
    v6 = sub_BCD140(*(_QWORD **)(a3 + 72), 1u);
    v7 = sub_ACD640(v6, 0, 0);
    v50 = 0;
    v47 = (_BYTE *)v7;
    if ( *(_DWORD *)(v49 + 12) )
    {
      v8 = (_BYTE *)v7;
      while ( 1 )
      {
        v55 = 257;
        v9 = sub_94D3D0((unsigned int **)a3, v48, (__int64)&v50, 1, (__int64)v54);
        v53 = 257;
        v10 = *(_QWORD *)(v9 + 8);
        v11 = v9;
        if ( *(_BYTE *)(v10 + 8) != 12 )
        {
          v12 = v9;
          do
          {
            v13 = sub_24650D0(v4, v12, a3);
            v10 = *(_QWORD *)(v13 + 8);
            v12 = v13;
          }
          while ( *(_BYTE *)(v10 + 8) != 12 );
          v11 = v13;
        }
        v14 = (_BYTE *)v11;
        if ( *(_DWORD *)(v10 + 8) >> 8 == 1 )
          goto LABEL_14;
        v15 = sub_AD64C0(v10, 0, 0);
        v16 = *(_QWORD *)(a3 + 80);
        v17 = (unsigned __int8 *)v15;
        v18 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v16 + 56LL);
        if ( v18 == sub_928890 )
        {
          if ( *(_BYTE *)v11 > 0x15u || *v17 > 0x15u )
          {
LABEL_19:
            v44 = (__int64)v17;
            v55 = 257;
            v21 = sub_BD2C40(72, unk_3F10FD0);
            v22 = v44;
            v14 = v21;
            if ( v21 )
            {
              v23 = *(_QWORD *)(v11 + 8);
              v24 = *(unsigned __int8 *)(v23 + 8);
              if ( (unsigned int)(v24 - 17) > 1 )
              {
                v28 = sub_BCB2A0(*(_QWORD **)v23);
              }
              else
              {
                v25 = *(_DWORD *)(v23 + 32);
                v26 = *(_QWORD **)v23;
                BYTE4(v51) = (_BYTE)v24 == 18;
                LODWORD(v51) = v25;
                v27 = (__int64 *)sub_BCB2A0(v26);
                v28 = sub_BCE1B0(v27, v51);
              }
              sub_B523C0((__int64)v14, v28, 53, 33, v11, v44, (__int64)v54, 0, 0, 0);
            }
            (*(void (__fastcall **)(_QWORD, _BYTE *, int *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a3 + 88) + 16LL))(
              *(_QWORD *)(a3 + 88),
              v14,
              v52,
              *(_QWORD *)(a3 + 56),
              *(_QWORD *)(a3 + 64),
              v22);
            if ( *(_QWORD *)a3 != *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8) )
            {
              v45 = v4;
              v29 = *(unsigned int **)a3;
              v30 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
              do
              {
                v31 = *((_QWORD *)v29 + 1);
                v32 = *v29;
                v29 += 4;
                sub_B99FD0((__int64)v14, v32, v31);
              }
              while ( (unsigned int *)v30 != v29 );
              v4 = v45;
            }
            goto LABEL_14;
          }
          v43 = v17;
          v19 = sub_AAB310(0x21u, (unsigned __int8 *)v11, v17);
          v17 = v43;
          v14 = (_BYTE *)v19;
        }
        else
        {
          v46 = v17;
          v33 = v18(v16, 33u, (_BYTE *)v11, v17);
          v17 = v46;
          v14 = (_BYTE *)v33;
        }
        if ( !v14 )
          goto LABEL_19;
LABEL_14:
        if ( v8 == v47 )
        {
          v8 = v14;
        }
        else
        {
          v55 = 257;
          v8 = (_BYTE *)sub_A82480((unsigned int **)a3, v8, v14, (__int64)v54);
        }
        if ( *(_DWORD *)(v49 + 12) <= (unsigned int)++v50 )
          return (unsigned __int64)v8;
      }
    }
    return v7;
  }
  else
  {
    while ( v5 != 16 )
    {
      if ( (unsigned int)v5 - 17 > 1 )
        return v48;
      if ( v5 != 18 )
      {
        v54[0] = sub_BCAE30(v49);
        v55 = 257;
        v40 = *(_QWORD *)(a1 + 8);
        v54[1] = v41;
        v42 = (__int64 **)sub_BCCE00(*(_QWORD **)(v40 + 72), v54[0]);
        return sub_24633A0((__int64 *)a3, 0x31u, v48, v42, (__int64)v54, 0, v52[0], 0);
      }
      v48 = sub_B34870(a3, v48);
      v49 = *(_QWORD *)(v48 + 8);
      v5 = *(_BYTE *)(v49 + 8);
      if ( v5 == 15 )
        goto LABEL_2;
    }
    if ( !*(_QWORD *)(v49 + 32) )
    {
      v39 = sub_BCD140(*(_QWORD **)(a3 + 72), 1u);
      return sub_ACD640(v39, 0, 0);
    }
    v52[0] = 0;
    v55 = 257;
    v34 = sub_94D3D0((unsigned int **)a3, v48, (__int64)v52, 1, (__int64)v54);
    v35 = sub_24650D0(a1, v34, a3);
    v52[0] = 1;
    v8 = (_BYTE *)v35;
    if ( *(_QWORD *)(v49 + 32) > 1u )
    {
      v36 = (_BYTE *)v35;
      do
      {
        v55 = 257;
        v37 = sub_94D3D0((unsigned int **)a3, v48, (__int64)v52, 1, (__int64)v54);
        v38 = (_BYTE *)sub_24650D0(a1, v37, a3);
        v55 = 257;
        v36 = (_BYTE *)sub_A82480((unsigned int **)a3, v36, v38, (__int64)v54);
        ++v52[0];
      }
      while ( (unsigned __int64)(unsigned int)v52[0] < *(_QWORD *)(v49 + 32) );
      return (unsigned __int64)v36;
    }
  }
  return (unsigned __int64)v8;
}
