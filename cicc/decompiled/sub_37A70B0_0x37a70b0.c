// Function: sub_37A70B0
// Address: 0x37a70b0
//
unsigned __int8 *__fastcall sub_37A70B0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rax
  __int16 v5; // dx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // r14
  unsigned __int16 *v12; // rdx
  int v13; // eax
  __int64 v14; // rdx
  unsigned __int16 *v15; // rdx
  unsigned __int64 v16; // r10
  unsigned __int64 v17; // r8
  __int64 v18; // rsi
  unsigned __int16 v19; // ax
  __int64 v20; // rdx
  unsigned int v21; // r13d
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int128 v28; // rax
  __int64 v29; // r14
  __int64 v30; // r15
  __int64 (__fastcall *v31)(__int64, __int64, __int64, __int64); // rbx
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 (__fastcall *v34)(__int64, __int64, unsigned int); // rax
  int v35; // edx
  unsigned __int16 v36; // ax
  __int128 v37; // rax
  __int64 v38; // r9
  unsigned __int8 *v39; // r12
  __int64 v40; // rdx
  __int128 v42; // [rsp-20h] [rbp-B0h]
  char v43; // [rsp+7h] [rbp-89h]
  char v44; // [rsp+7h] [rbp-89h]
  unsigned int v45; // [rsp+8h] [rbp-88h]
  __int64 v46; // [rsp+8h] [rbp-88h]
  int v47; // [rsp+8h] [rbp-88h]
  unsigned __int8 v48; // [rsp+10h] [rbp-80h]
  __int64 *v49; // [rsp+10h] [rbp-80h]
  __int128 v50; // [rsp+10h] [rbp-80h]
  char v51; // [rsp+10h] [rbp-80h]
  __int64 v52; // [rsp+28h] [rbp-68h]
  unsigned int v53; // [rsp+30h] [rbp-60h] BYREF
  __int64 v54; // [rsp+38h] [rbp-58h]
  __int16 v55; // [rsp+40h] [rbp-50h] BYREF
  __int64 v56; // [rsp+48h] [rbp-48h]
  __int64 v57; // [rsp+50h] [rbp-40h] BYREF
  int v58; // [rsp+58h] [rbp-38h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v54 = *((_QWORD *)v4 + 1);
  v6 = *(_QWORD *)(a2 + 40);
  LOWORD(v53) = v5;
  v7 = sub_379AB60((__int64)a1, *(_QWORD *)v6, *(_QWORD *)(v6 + 8));
  v10 = v9;
  v11 = v7;
  v12 = (unsigned __int16 *)(*(_QWORD *)(v7 + 48) + 16LL * (unsigned int)v9);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  v55 = v13;
  v56 = v14;
  if ( (_WORD)v13 )
  {
    v15 = word_4456340;
    LOBYTE(v8) = (unsigned __int16)(v13 - 176) <= 0x34u;
    LOBYTE(v16) = v8;
    v17 = word_4456340[v13 - 1];
  }
  else
  {
    v17 = sub_3007240((__int64)&v55);
    v16 = HIDWORD(v17);
    v8 = HIDWORD(v17);
  }
  v18 = *(_QWORD *)(a2 + 80);
  v57 = v18;
  if ( v18 )
  {
    v48 = v8;
    v43 = v16;
    v45 = v17;
    sub_B96E90((__int64)&v57, v18, 1);
    v17 = v45;
    v8 = v48;
    LOBYTE(v16) = v43;
  }
  v58 = *(_DWORD *)(a2 + 72);
  if ( (_WORD)v53 )
  {
    v19 = word_4456580[(unsigned __int16)v53 - 1];
    v20 = 0;
  }
  else
  {
    v44 = v16;
    v47 = v17;
    v51 = v8;
    v19 = sub_3009970((__int64)&v53, v18, (__int64)v15, v8, v17);
    LODWORD(v17) = v47;
    LOBYTE(v8) = v51;
    LOBYTE(v16) = v44;
  }
  LODWORD(v52) = v17;
  v21 = v19;
  BYTE4(v52) = v16;
  v46 = v20;
  v49 = *(__int64 **)(a1[1] + 64);
  if ( (_BYTE)v8 )
    LOWORD(v22) = sub_2D43AD0(v19, v17);
  else
    LOWORD(v22) = sub_2D43050(v19, v17);
  if ( (_WORD)v22 )
  {
    v25 = *a1;
    v26 = (unsigned __int16)v22;
    v27 = 0;
  }
  else
  {
    v22 = sub_3009450(v49, v21, v46, v52, v23, v24);
    v26 = v22;
    v27 = v40;
    if ( !(_WORD)v22 )
      goto LABEL_24;
    v25 = *a1;
  }
  if ( !*(_QWORD *)(v25 + 8LL * (unsigned __int16)v22 + 112) )
  {
LABEL_24:
    v39 = sub_3412A00((_QWORD *)a1[1], a2, 0, v26, v27, v24, a3);
    goto LABEL_25;
  }
  LOWORD(v26) = v22;
  *((_QWORD *)&v42 + 1) = v10;
  *(_QWORD *)&v42 = v11;
  *(_QWORD *)&v28 = sub_3406EB0(
                      (_QWORD *)a1[1],
                      *(_DWORD *)(a2 + 24),
                      (__int64)&v57,
                      v26,
                      v27,
                      v24,
                      v42,
                      *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  v29 = a1[1];
  v30 = *a1;
  v50 = v28;
  v31 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 72LL);
  v32 = sub_2E79000(*(__int64 **)(v29 + 40));
  v33 = v32;
  if ( v31 == sub_2FE4D20 )
  {
    v34 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v30 + 32LL);
    if ( v34 == sub_2D42F30 )
    {
      v35 = sub_AE2980(v33, 0)[1];
      v36 = 2;
      if ( v35 != 1 )
      {
        v36 = 3;
        if ( v35 != 2 )
        {
          v36 = 4;
          if ( v35 != 4 )
          {
            v36 = 5;
            if ( v35 != 8 )
            {
              v36 = 6;
              if ( v35 != 16 )
              {
                v36 = 7;
                if ( v35 != 32 )
                {
                  v36 = 8;
                  if ( v35 != 64 )
                    v36 = 9 * (v35 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v36 = v34(v30, v33, 0);
    }
  }
  else
  {
    v36 = ((__int64 (__fastcall *)(__int64, __int64))v31)(v30, v32);
  }
  *(_QWORD *)&v37 = sub_3400BD0(v29, 0, (__int64)&v57, v36, 0, 0, a3, 0);
  v39 = sub_3406EB0((_QWORD *)v29, 0xA1u, (__int64)&v57, v53, v54, v38, v50, v37);
LABEL_25:
  if ( v57 )
    sub_B91220((__int64)&v57, v57);
  return v39;
}
