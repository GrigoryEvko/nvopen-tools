// Function: sub_337DE80
// Address: 0x337de80
//
void __fastcall sub_337DE80(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 (*v6)(); // rdx
  __int64 (*v7)(); // rax
  __int64 v8; // r14
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rsi
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // r14
  int v15; // r9d
  __int64 (__fastcall *v16)(__int64, __int64, unsigned int); // rbx
  __int64 v17; // rdi
  int v18; // edx
  unsigned __int16 v19; // ax
  unsigned int v20; // ebx
  int v21; // eax
  int v22; // edx
  __int64 v23; // rdx
  int v24; // r9d
  __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // edx
  __int64 (__fastcall *v28)(__int64, __int64, unsigned int); // rbx
  __int64 v29; // rdi
  int v30; // edx
  unsigned __int16 v31; // ax
  int v32; // edx
  __int64 v33; // r14
  unsigned int *v34; // rbx
  __int64 v35; // rax
  int v36; // edx
  unsigned __int16 v37; // ax
  int v38; // eax
  int v39; // edx
  int v40; // r12d
  __int64 v41; // rdx
  int v42; // r9d
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // r12
  int v47; // edx
  int v48; // eax
  int v49; // edx
  int v50; // r9d
  __int64 v51; // rbx
  int v52; // edx
  _QWORD *v53; // rax
  __int64 v54; // rsi
  __int128 v55; // [rsp-20h] [rbp-160h]
  __int128 v56; // [rsp-10h] [rbp-150h]
  __int128 v57; // [rsp-10h] [rbp-150h]
  __int128 v58; // [rsp-10h] [rbp-150h]
  __int64 v59; // [rsp+8h] [rbp-138h]
  int v60; // [rsp+10h] [rbp-130h]
  unsigned int v61; // [rsp+10h] [rbp-130h]
  int v62; // [rsp+18h] [rbp-128h]
  unsigned int v63; // [rsp+18h] [rbp-128h]
  int v64; // [rsp+20h] [rbp-120h]
  unsigned int *v65; // [rsp+30h] [rbp-110h]
  __int64 (__fastcall *v66)(__int64, __int64, unsigned int); // [rsp+30h] [rbp-110h]
  __int64 v67; // [rsp+30h] [rbp-110h]
  unsigned int v68; // [rsp+40h] [rbp-100h]
  __int64 v70; // [rsp+90h] [rbp-B0h] BYREF
  int v71; // [rsp+98h] [rbp-A8h]
  __int128 v72; // [rsp+A0h] [rbp-A0h] BYREF
  __int128 v73; // [rsp+B0h] [rbp-90h]
  __int64 v74; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v75; // [rsp+C8h] [rbp-78h]
  __int64 v76; // [rsp+D0h] [rbp-70h]
  __int64 v77; // [rsp+D8h] [rbp-68h]
  unsigned __int64 v78[2]; // [rsp+E0h] [rbp-60h] BYREF
  _BYTE v79[80]; // [rsp+F0h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a1[108] + 16);
  v4 = sub_B2E500(*(_QWORD *)a1[120]);
  v5 = *(_QWORD *)v3;
  v6 = *(__int64 (**)())(*(_QWORD *)v3 + 872LL);
  if ( v6 == sub_2E2F9C0 )
  {
LABEL_2:
    v7 = *(__int64 (**)())(v5 + 880);
    if ( v7 == sub_2E2F9D0 || !((unsigned int (__fastcall *)(__int64, __int64))v7)(v3, v4) )
      return;
    goto LABEL_5;
  }
  if ( !((unsigned int (__fastcall *)(__int64, __int64))v6)(v3, v4) )
  {
    v5 = *(_QWORD *)v3;
    goto LABEL_2;
  }
LABEL_5:
  v8 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v8 + 8) != 11 )
  {
    v9 = *((_DWORD *)a1 + 212);
    v70 = 0;
    v78[0] = (unsigned __int64)v79;
    v78[1] = 0x200000000LL;
    v10 = *a1;
    v71 = v9;
    if ( v10 )
    {
      if ( &v70 != (__int64 *)(v10 + 48) )
      {
        v11 = *(_QWORD *)(v10 + 48);
        v70 = v11;
        if ( v11 )
        {
          sub_B96E90((__int64)&v70, v11, 1);
          v8 = *(_QWORD *)(a2 + 8);
        }
      }
    }
    v12 = sub_2E79000(*(__int64 **)(a1[108] + 40));
    LOBYTE(v75) = 0;
    *((_QWORD *)&v56 + 1) = v75;
    v74 = 0;
    *(_QWORD *)&v56 = 0;
    sub_34B8C80(v3, v12, v8, (unsigned int)v78, 0, 0, v56);
    v13 = a1[120];
    v14 = a1[108];
    v72 = 0;
    v15 = *(_DWORD *)(v13 + 884);
    v73 = 0;
    if ( v15 )
    {
      v65 = (unsigned int *)v78[0];
      v16 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v3 + 32LL);
      v17 = sub_2E79000(*(__int64 **)(v14 + 40));
      if ( v16 == sub_2D42F30 )
      {
        v18 = sub_AE2980(v17, 0)[1];
        v19 = 2;
        if ( v18 != 1 )
        {
          v19 = 3;
          if ( v18 != 2 )
          {
            v19 = 4;
            if ( v18 != 4 )
            {
              v19 = 5;
              if ( v18 != 8 )
              {
                v19 = 6;
                if ( v18 != 16 )
                {
                  v19 = 7;
                  if ( v18 != 32 )
                  {
                    v19 = 8;
                    if ( v18 != 64 )
                      v19 = 9 * (v18 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v19 = v16(v3, v17, 0);
      }
      v20 = v19;
      v59 = a1[108] + 288;
      v68 = *(_DWORD *)(a1[120] + 884);
      v21 = sub_33E5110(v14, v19, 0, 1, 0);
      v62 = v22;
      v60 = v21;
      v74 = v59;
      LODWORD(v75) = 0;
      v76 = sub_33F0B60(v14, v68, v20, 0);
      v77 = v23;
      *((_QWORD *)&v57 + 1) = 2;
      *(_QWORD *)&v57 = &v74;
      v25 = sub_3411630(v14, 50, (unsigned int)&v70, v60, v62, v24, v57);
      *(_QWORD *)&v72 = sub_33FB310(v14, v25, v26, &v70, *v65, *((_QWORD *)v65 + 1));
      DWORD2(v72) = v27;
    }
    else
    {
      v28 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v3 + 32LL);
      v29 = sub_2E79000(*(__int64 **)(v14 + 40));
      if ( v28 == sub_2D42F30 )
      {
        v30 = sub_AE2980(v29, 0)[1];
        v31 = 2;
        if ( v30 != 1 )
        {
          v31 = 3;
          if ( v30 != 2 )
          {
            v31 = 4;
            if ( v30 != 4 )
            {
              v31 = 5;
              if ( v30 != 8 )
              {
                v31 = 6;
                if ( v30 != 16 )
                {
                  v31 = 7;
                  if ( v30 != 32 )
                  {
                    v31 = 8;
                    if ( v30 != 64 )
                      v31 = 9 * (v30 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v31 = v28(v3, v29, 0);
      }
      *(_QWORD *)&v72 = sub_3400BD0(v14, 0, (unsigned int)&v70, v31, 0, 0, 0);
      DWORD2(v72) = v32;
    }
    v33 = a1[108];
    v34 = (unsigned int *)(v78[0] + 16);
    v66 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v3 + 32LL);
    v35 = sub_2E79000(*(__int64 **)(v33 + 40));
    if ( v66 == sub_2D42F30 )
    {
      v36 = sub_AE2980(v35, 0)[1];
      v37 = 2;
      if ( v36 != 1 )
      {
        v37 = 3;
        if ( v36 != 2 )
        {
          v37 = 4;
          if ( v36 != 4 )
          {
            v37 = 5;
            if ( v36 != 8 )
            {
              v37 = 6;
              if ( v36 != 16 )
              {
                v37 = 7;
                if ( v36 != 32 )
                {
                  v37 = 8;
                  if ( v36 != 64 )
                    v37 = 9 * (v36 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v37 = v66(v3, v35, 0);
    }
    v63 = v37;
    v67 = a1[108] + 288;
    v61 = *(_DWORD *)(a1[120] + 888);
    v38 = sub_33E5110(v33, v37, 0, 1, 0);
    v64 = v39;
    v40 = v38;
    LODWORD(v75) = 0;
    v74 = v67;
    v76 = sub_33F0B60(v33, v61, v63, 0);
    v77 = v41;
    *((_QWORD *)&v58 + 1) = 2;
    *(_QWORD *)&v58 = &v74;
    v43 = sub_3411630(v33, 50, (unsigned int)&v70, v40, v64, v42, v58);
    v45 = sub_33FB310(v33, v43, v44, &v70, *v34, *((_QWORD *)v34 + 1));
    v46 = a1[108];
    *(_QWORD *)&v73 = v45;
    DWORD2(v73) = v47;
    v48 = sub_33E5830(v46, v78[0]);
    *((_QWORD *)&v55 + 1) = 2;
    *(_QWORD *)&v55 = &v72;
    v51 = sub_3411630(v46, 55, (unsigned int)&v70, v48, v49, v50, v55);
    LODWORD(v46) = v52;
    v74 = a2;
    v53 = sub_337DC20((__int64)(a1 + 1), &v74);
    *v53 = v51;
    v54 = v70;
    *((_DWORD *)v53 + 2) = v46;
    if ( v54 )
      sub_B91220((__int64)&v70, v54);
    if ( (_BYTE *)v78[0] != v79 )
      _libc_free(v78[0]);
  }
}
