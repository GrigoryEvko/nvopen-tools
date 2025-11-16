// Function: sub_1FDE000
// Address: 0x1fde000
//
__int64 __fastcall sub_1FDE000(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // rax
  __int64 v5; // rax
  unsigned int v6; // r15d
  unsigned __int8 v7; // al
  __int64 v8; // r14
  __int64 v9; // r13
  unsigned __int8 v10; // bl
  int v11; // r14d
  int i; // r13d
  unsigned int v13; // eax
  __int64 v15; // rbx
  int v16; // eax
  __int64 v17; // r13
  int v18; // r14d
  __int64 v19; // rax
  __int64 v20; // r13
  char v21; // di
  unsigned int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-158h]
  __int64 v31; // [rsp+10h] [rbp-150h]
  __int64 v32; // [rsp+28h] [rbp-138h]
  __int64 v33; // [rsp+30h] [rbp-130h]
  __int64 v34; // [rsp+38h] [rbp-128h]
  __int64 v35; // [rsp+50h] [rbp-110h]
  __int64 v36; // [rsp+58h] [rbp-108h]
  unsigned __int8 v37; // [rsp+6Bh] [rbp-F5h] BYREF
  int v38; // [rsp+6Ch] [rbp-F4h] BYREF
  __int64 v39; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v40; // [rsp+78h] [rbp-E8h]
  __int64 v41; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v42; // [rsp+88h] [rbp-D8h]
  __int64 v43; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v44; // [rsp+98h] [rbp-C8h]
  __int64 v45; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v46; // [rsp+A8h] [rbp-B8h]
  __int64 v47; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v48; // [rsp+B8h] [rbp-A8h]
  _BYTE v49[8]; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v50; // [rsp+C8h] [rbp-98h]
  __int64 v51; // [rsp+D0h] [rbp-90h]
  _BYTE *v52; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v53; // [rsp+E8h] [rbp-78h]
  _BYTE v54[112]; // [rsp+F0h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a1 + 8);
  v35 = 0;
  v4 = *(__int64 (**)())(**(_QWORD **)(v3 + 16) + 56LL);
  if ( v4 != sub_1D12D20 )
  {
    v28 = ((__int64 (__fastcall *)(_QWORD))v4)(*(_QWORD *)(v3 + 16));
    v3 = *(_QWORD *)(a1 + 8);
    v35 = v28;
  }
  v52 = v54;
  v53 = 0x400000000LL;
  v5 = sub_1E0A0C0(v3);
  sub_20C7CE0(v35, v5, a2, &v52, 0, 0);
  v6 = v53;
  if ( (_DWORD)v53 )
  {
    v6 = 0;
    v36 = 0;
    v34 = 16LL * (unsigned int)v53;
    do
    {
      v7 = v52[v36];
      v8 = *(_QWORD *)&v52[v36 + 8];
      LOBYTE(v39) = v7;
      v40 = v8;
      v9 = *a2;
      if ( v7 )
      {
        v10 = *(_BYTE *)(v35 + v7 + 1155);
        v11 = *(unsigned __int8 *)(v35 + v7 + 1040);
        goto LABEL_7;
      }
      if ( sub_1F58D20((__int64)&v39) )
      {
        v49[0] = 0;
        v50 = 0;
        LOBYTE(v45) = 0;
        sub_1F426C0(v35, v9, (unsigned int)v39, v8, (__int64)v49, (unsigned int *)&v47, &v45);
        v10 = v45;
        v32 = *a2;
        goto LABEL_20;
      }
      sub_1F40D10((__int64)v49, v35, v9, v39, v40);
      v15 = v51;
      LOBYTE(v41) = v50;
      v42 = v51;
      if ( (_BYTE)v50 )
      {
        v10 = *(_BYTE *)(v35 + (unsigned __int8)v50 + 1155);
      }
      else
      {
        if ( sub_1F58D20((__int64)&v41) )
        {
          v49[0] = 0;
          v50 = 0;
          LOBYTE(v45) = 0;
          sub_1F426C0(v35, v9, (unsigned int)v41, v15, (__int64)v49, (unsigned int *)&v47, &v45);
LABEL_40:
          v10 = v45;
          goto LABEL_19;
        }
        sub_1F40D10((__int64)v49, v35, v9, v41, v42);
        v23 = (unsigned __int8)v50;
        v24 = v51;
        LOBYTE(v43) = v50;
        v44 = v51;
        if ( !(_BYTE)v50 )
        {
          if ( sub_1F58D20((__int64)&v43) )
          {
            v49[0] = 0;
            v50 = 0;
            LOBYTE(v45) = 0;
            sub_1F426C0(v35, v9, (unsigned int)v43, v24, (__int64)v49, (unsigned int *)&v47, &v45);
            goto LABEL_40;
          }
          sub_1F40D10((__int64)v49, v35, v9, v43, v44);
          v23 = (unsigned __int8)v50;
          v25 = v51;
          LOBYTE(v45) = v50;
          v46 = v51;
          if ( !(_BYTE)v50 )
          {
            if ( sub_1F58D20((__int64)&v45) )
            {
              v49[0] = 0;
              v50 = 0;
              LOBYTE(v38) = 0;
              sub_1F426C0(v35, v9, (unsigned int)v45, v25, (__int64)v49, (unsigned int *)&v47, &v38);
              v10 = v38;
            }
            else
            {
              sub_1F40D10((__int64)v49, v35, v9, v45, v46);
              LOBYTE(v47) = v50;
              v48 = v51;
              if ( (_BYTE)v50 )
              {
                v10 = *(_BYTE *)(v35 + (unsigned __int8)v50 + 1155);
              }
              else if ( sub_1F58D20((__int64)&v47) )
              {
                v49[0] = 0;
                v50 = 0;
                v37 = 0;
                sub_1F426C0(v35, v9, (unsigned int)v47, v48, (__int64)v49, (unsigned int *)&v38, &v37);
                v10 = v37;
              }
              else
              {
                sub_1F40D10((__int64)v49, v35, v9, v47, v48);
                v29 = v30;
                LOBYTE(v29) = v50;
                v30 = v29;
                v10 = sub_1D5E9F0(v35, v9, (unsigned int)v29, v51);
              }
            }
            goto LABEL_19;
          }
        }
        v10 = *(_BYTE *)(v35 + v23 + 1155);
      }
LABEL_19:
      v32 = *a2;
LABEL_20:
      LOBYTE(v41) = 0;
      v42 = v8;
      if ( !sub_1F58D20((__int64)&v41) )
      {
        v16 = sub_1F58D40((__int64)&v41);
        v17 = v42;
        v18 = v16;
        v43 = v41;
        v33 = v41;
        v44 = v42;
        if ( sub_1F58D20((__int64)&v43) )
        {
          v49[0] = 0;
          v50 = 0;
          LOBYTE(v45) = 0;
          sub_1F426C0(v35, v32, (unsigned int)v43, v17, (__int64)v49, (unsigned int *)&v47, &v45);
          v21 = v45;
        }
        else
        {
          sub_1F40D10((__int64)v49, v35, v32, v33, v17);
          v19 = (unsigned __int8)v50;
          v20 = v51;
          LOBYTE(v45) = v50;
          v46 = v51;
          if ( (_BYTE)v50 )
            goto LABEL_24;
          if ( sub_1F58D20((__int64)&v45) )
          {
            v49[0] = 0;
            v50 = 0;
            LOBYTE(v39) = 0;
            sub_1F426C0(v35, v32, (unsigned int)v45, v20, (__int64)v49, (unsigned int *)&v47, &v39);
            v21 = v39;
          }
          else
          {
            sub_1F40D10((__int64)v49, v35, v32, v45, v46);
            v19 = (unsigned __int8)v50;
            v26 = v51;
            LOBYTE(v47) = v50;
            v48 = v51;
            if ( (_BYTE)v50 )
            {
LABEL_24:
              v21 = *(_BYTE *)(v35 + v19 + 1155);
            }
            else if ( sub_1F58D20((__int64)&v47) )
            {
              v49[0] = 0;
              v50 = 0;
              LOBYTE(v38) = 0;
              sub_1F426C0(v35, v32, (unsigned int)v47, v26, (__int64)v49, (unsigned int *)&v39, &v38);
              v21 = v38;
            }
            else
            {
              sub_1F40D10((__int64)v49, v35, v32, v47, v48);
              v27 = v31;
              LOBYTE(v27) = v50;
              v31 = v27;
              v21 = sub_1D5E9F0(v35, v32, (unsigned int)v27, v51);
            }
          }
        }
        v22 = sub_1FDDC20(v21);
        v11 = (v22 + v18 - 1) / v22;
        goto LABEL_7;
      }
      v49[0] = 0;
      v50 = 0;
      LOBYTE(v45) = 0;
      v11 = sub_1F426C0(v35, v32, (unsigned int)v41, v8, (__int64)v49, (unsigned int *)&v47, &v45);
LABEL_7:
      if ( v11 )
      {
        for ( i = 0; i != v11; ++i )
        {
          v13 = sub_1FDDF90(a1, v10);
          if ( !v6 )
            v6 = v13;
        }
      }
      v36 += 16;
    }
    while ( v34 != v36 );
  }
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  return v6;
}
