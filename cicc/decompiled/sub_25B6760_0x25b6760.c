// Function: sub_25B6760
// Address: 0x25b6760
//
__int64 __fastcall sub_25B6760(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v6; // rax
  __int64 *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // r12
  __int64 v18; // rbx
  char v19; // dl
  __int64 v20; // rax
  _QWORD *v21; // rdi
  __int64 *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r14
  int v26; // r11d
  char v27; // dl
  unsigned int v28; // ebx
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // rdx
  unsigned __int64 v35; // r15
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // r14
  __int64 v44; // rdx
  __int64 v45; // rcx
  char v46; // bl
  __int64 v47; // rax
  __int64 v48; // r14
  __int64 v49; // rax
  char v50; // al
  __int16 v51; // cx
  _QWORD *v52; // rax
  __int64 v53; // r9
  __int64 v54; // rbx
  unsigned int *v55; // r12
  __int64 v56; // r14
  __int64 v57; // rdx
  __int64 *v58; // rdi
  unsigned __int16 v59; // [rsp+Ch] [rbp-C4h]
  unsigned __int16 v60; // [rsp+Eh] [rbp-C2h]
  _QWORD *v61; // [rsp+18h] [rbp-B8h]
  __int16 v62; // [rsp+20h] [rbp-B0h]
  _QWORD *v63; // [rsp+20h] [rbp-B0h]
  __int64 v64; // [rsp+28h] [rbp-A8h]
  int v65; // [rsp+28h] [rbp-A8h]
  char v67; // [rsp+30h] [rbp-A0h]
  int v68; // [rsp+30h] [rbp-A0h]
  unsigned __int8 v70; // [rsp+40h] [rbp-90h]
  __int64 v71; // [rsp+40h] [rbp-90h]
  __int64 v72; // [rsp+40h] [rbp-90h]
  unsigned int v73; // [rsp+40h] [rbp-90h]
  __int64 v74; // [rsp+48h] [rbp-88h]
  __int64 v75; // [rsp+50h] [rbp-80h] BYREF
  __int64 v76; // [rsp+58h] [rbp-78h]
  _QWORD v77[2]; // [rsp+60h] [rbp-70h] BYREF
  _QWORD v78[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v79; // [rsp+90h] [rbp-40h]

  v6 = sub_BCE3C0(*(__int64 **)a2, a4);
  v7 = *(__int64 **)a2;
  v64 = v6;
  v78[0] = v6;
  v8 = sub_B6DC00((__int64)v7, 375, (__int64)v78);
  v9 = 375;
  v10 = sub_B6E100((__int64 *)a2, 0x177u, (__int64)v78, 1, v8);
  v70 = 0;
  v61 = (_QWORD *)v10;
  if ( v10 )
  {
    v11 = *(_QWORD *)(v10 + 16);
    if ( !v11 )
      goto LABEL_48;
    do
    {
      v12 = v11;
      v11 = *(_QWORD *)(v11 + 8);
      v13 = *(_QWORD *)(v12 + 24);
      if ( *(_BYTE *)v13 == 85 )
      {
        v42 = *(_QWORD *)(v13 - 32);
        if ( v42 )
        {
          if ( !*(_BYTE *)v42
            && *(_QWORD *)(v42 + 24) == *(_QWORD *)(v13 + 80)
            && (*(_BYTE *)(v42 + 33) & 0x20) != 0
            && *(_DWORD *)(v42 + 36) == 375 )
          {
            v43 = sub_B43CB0(v13);
            if ( !(*(_DWORD *)(*(_QWORD *)(v43 + 24) + 8LL) >> 8) )
            {
              v46 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 176) + 8LL))(*(_QWORD *)(a1 + 176));
              v47 = (unsigned int)(*(_DWORD *)(v43 + 104) - 1);
              if ( (*(_BYTE *)(v43 + 2) & 1) != 0 )
              {
                v73 = *(_DWORD *)(v43 + 104) - 1;
                sub_B2C6D0(v43, v9, v44, v45);
                v47 = v73;
              }
              v48 = *(_QWORD *)(v43 + 96) + 40 * v47;
              v71 = *(_QWORD *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
              sub_D5F1F0(a3, v13);
              if ( v46 )
              {
                v49 = sub_AA4E30(*(_QWORD *)(a3 + 48));
                v50 = sub_AE5020(v49, *(_QWORD *)(v48 + 8));
                HIBYTE(v51) = HIBYTE(v62);
                v79 = 257;
                LOBYTE(v51) = v50;
                v62 = v51;
                v52 = sub_BD2C40(80, unk_3F10A10);
                v54 = (__int64)v52;
                if ( v52 )
                  sub_B4D3C0((__int64)v52, v48, v71, 0, v62, v53, 0, 0);
                v9 = v54;
                (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
                  *(_QWORD *)(a3 + 88),
                  v54,
                  v78,
                  *(_QWORD *)(a3 + 56),
                  *(_QWORD *)(a3 + 64));
                if ( *(_QWORD *)a3 != *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8) )
                {
                  v72 = v13;
                  v55 = *(unsigned int **)a3;
                  v56 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
                  do
                  {
                    v57 = *((_QWORD *)v55 + 1);
                    v9 = *v55;
                    v55 += 4;
                    sub_B99FD0(v54, v9, v57);
                  }
                  while ( (unsigned int *)v56 != v55 );
                  v13 = v72;
                }
              }
              else
              {
                v58 = *(__int64 **)(a3 + 72);
                HIDWORD(v76) = 0;
                v79 = 257;
                v77[1] = v48;
                v77[0] = v71;
                v9 = 373;
                v75 = sub_BCE3C0(v58, *(_DWORD *)(a2 + 316));
                sub_B33D10(a3, 0x175u, (__int64)&v75, 1, (int)v77, 2, v76, (__int64)v78);
              }
              sub_B43D60((_QWORD *)v13);
              v70 = 1;
            }
          }
        }
      }
    }
    while ( v11 );
    if ( !v61[2] )
LABEL_48:
      sub_B2E860(v61);
  }
  v14 = *(__int64 **)a2;
  v78[0] = v64;
  v15 = sub_B6DC00((__int64)v14, 374, (__int64)v78);
  v16 = sub_B6E100((__int64 *)a2, 0x176u, (__int64)v78, 1, v15);
  v17 = (_QWORD *)v16;
  if ( v16 )
  {
    v18 = *(_QWORD *)(v16 + 16);
    v19 = 0;
    if ( !v18 )
      goto LABEL_49;
    do
    {
      v20 = v18;
      v18 = *(_QWORD *)(v18 + 8);
      v21 = *(_QWORD **)(v20 + 24);
      if ( *(_BYTE *)v21 == 85 )
      {
        v41 = *(v21 - 4);
        if ( v41 )
        {
          if ( !*(_BYTE *)v41
            && *(_QWORD *)(v41 + 24) == v21[10]
            && (*(_BYTE *)(v41 + 33) & 0x20) != 0
            && *(_DWORD *)(v41 + 36) == 374 )
          {
            sub_B43D60(v21);
            v19 = 1;
          }
        }
      }
    }
    while ( v18 );
    v70 |= v19;
    if ( !v17[2] )
LABEL_49:
      sub_B2E860(v17);
  }
  v22 = *(__int64 **)a2;
  v78[0] = v64;
  v23 = sub_B6DC00((__int64)v22, 373, (__int64)v78);
  v24 = sub_B6E100((__int64 *)a2, 0x175u, (__int64)v78, 1, v23);
  v63 = (_QWORD *)v24;
  if ( v24 )
  {
    v25 = *(_QWORD *)(v24 + 16);
    if ( !v25 )
      goto LABEL_47;
    v26 = v59;
    v27 = 0;
    v74 = a2 + 312;
    v28 = v60;
    do
    {
      v29 = v25;
      v25 = *(_QWORD *)(v25 + 8);
      v30 = *(_QWORD *)(v29 + 24);
      if ( *(_BYTE *)v30 == 85 )
      {
        v32 = *(_QWORD *)(v30 - 32);
        if ( v32 )
        {
          if ( !*(_BYTE *)v32
            && *(_QWORD *)(v32 + 24) == *(_QWORD *)(v30 + 80)
            && (*(_BYTE *)(v32 + 33) & 0x20) != 0
            && *(_DWORD *)(v32 + 36) == 373 )
          {
            v65 = v26;
            BYTE1(v28) = 0;
            sub_D5F1F0(a3, v30);
            v33 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 176) + 16LL))(
                    *(_QWORD *)(a1 + 176),
                    *(_QWORD *)(a3 + 72));
            v67 = sub_AE5020(v74, v33);
            v78[0] = sub_9208B0(v74, v33);
            v78[1] = v34;
            v35 = (unsigned __int64)(v78[0] + 7LL) >> 3;
            v36 = sub_BCB2D0(*(_QWORD **)(a3 + 72));
            v37 = sub_ACD640(v36, (unsigned int)(((1LL << v67) + v35 - 1) >> v67 << v67), 0);
            v38 = *(_DWORD *)(v30 + 4) & 0x7FFFFFF;
            v39 = *(_QWORD *)(v30 + 32 * (1 - v38));
            v40 = *(_QWORD *)(v30 - 32 * v38);
            LODWORD(v38) = v65;
            BYTE1(v38) = 0;
            v68 = v38;
            sub_B343C0(a3, 0xEEu, v40, v38, v39, v28, v37, 0, 0, 0, 0, 0);
            sub_B43D60((_QWORD *)v30);
            v26 = v68;
            v27 = 1;
          }
        }
      }
    }
    while ( v25 );
    v70 |= v27;
    if ( !v63[2] )
LABEL_47:
      sub_B2E860(v63);
  }
  return v70;
}
