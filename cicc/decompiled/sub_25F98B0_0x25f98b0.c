// Function: sub_25F98B0
// Address: 0x25f98b0
//
void __fastcall sub_25F98B0(unsigned int *a1, unsigned int *a2)
{
  unsigned int *v2; // r13
  unsigned int v4; // ecx
  __int64 v5; // rdi
  __int64 v6; // r11
  __int64 v7; // r8
  __int64 v8; // rax
  int v9; // r15d
  int v10; // esi
  __int64 v11; // r12
  __int64 v12; // r11
  bool v13; // cf
  __int64 v14; // rbx
  __int64 v15; // rdx
  int v16; // r11d
  int v17; // r10d
  unsigned int v18; // r9d
  __int64 v19; // rdi
  __int64 v20; // r12
  unsigned __int64 v21; // rbx
  __int64 v22; // rsi
  unsigned int *v23; // rbx
  unsigned int v25; // [rsp+18h] [rbp-118h]
  int v26; // [rsp+1Ch] [rbp-114h]
  unsigned int v27; // [rsp+20h] [rbp-110h]
  int v28; // [rsp+24h] [rbp-10Ch]
  __int64 v29; // [rsp+28h] [rbp-108h]
  __int64 v30; // [rsp+30h] [rbp-100h]
  int v31; // [rsp+38h] [rbp-F8h]
  unsigned int v32; // [rsp+3Ch] [rbp-F4h]
  int v33; // [rsp+40h] [rbp-F0h]
  int v34; // [rsp+44h] [rbp-ECh]
  __int64 v35; // [rsp+48h] [rbp-E8h]
  __int64 v36; // [rsp+50h] [rbp-E0h]
  __int64 v37; // [rsp+58h] [rbp-D8h]
  _DWORD v38[2]; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v39; // [rsp+68h] [rbp-C8h]
  __int64 v40; // [rsp+70h] [rbp-C0h]
  __int64 v41; // [rsp+78h] [rbp-B8h]
  __int64 v42; // [rsp+80h] [rbp-B0h]
  int v43; // [rsp+88h] [rbp-A8h]
  int v44; // [rsp+8Ch] [rbp-A4h]
  unsigned int v45; // [rsp+90h] [rbp-A0h]
  __int64 v46; // [rsp+98h] [rbp-98h]
  __int64 v47; // [rsp+A0h] [rbp-90h]
  int v48; // [rsp+A8h] [rbp-88h]
  int v49; // [rsp+ACh] [rbp-84h]
  unsigned int v50; // [rsp+B0h] [rbp-80h]
  __int64 v51; // [rsp+B8h] [rbp-78h]
  __int64 v52; // [rsp+C0h] [rbp-70h]
  int v53; // [rsp+C8h] [rbp-68h]
  int v54; // [rsp+CCh] [rbp-64h]
  unsigned int v55; // [rsp+D0h] [rbp-60h]
  __int64 v56; // [rsp+D8h] [rbp-58h]
  __int64 v57; // [rsp+E0h] [rbp-50h]
  int v58; // [rsp+E8h] [rbp-48h]
  int v59; // [rsp+ECh] [rbp-44h]
  unsigned int v60; // [rsp+F0h] [rbp-40h]

  if ( a1 != a2 )
  {
    v2 = a1 + 38;
    while ( a2 != v2 )
    {
      v4 = *v2;
      v5 = (__int64)v2;
      v6 = *((_QWORD *)v2 + 11);
      v2 += 38;
      v37 = *((_QWORD *)v2 - 18);
      v36 = *((_QWORD *)v2 - 4) + 1LL;
      v35 = *((_QWORD *)v2 - 3);
      v34 = *(v2 - 4);
      v32 = *(v2 - 2);
      v30 = v6 + 1;
      v33 = *(v2 - 3);
      v31 = *(v2 - 12);
      v7 = *((_QWORD *)v2 - 11);
      v8 = *((_QWORD *)v2 - 12) + 1LL;
      v28 = *(v2 - 11);
      v9 = *(v2 - 37);
      v27 = *(v2 - 10);
      v10 = *(v2 - 20);
      v26 = *(v2 - 19);
      v11 = *((_QWORD *)v2 - 17);
      v25 = *(v2 - 18);
      v29 = *((_QWORD *)v2 - 7);
      v12 = *((_QWORD *)v2 - 16);
      v13 = v4 < *a1;
      v14 = *((_QWORD *)v2 - 15);
      v38[0] = v4;
      v15 = v12 + 1;
      v38[1] = v9;
      v16 = *(v2 - 28);
      v17 = *(v2 - 27);
      v18 = *(v2 - 26);
      if ( v13 )
      {
        v43 = *(v2 - 28);
        v19 = v5 - (_QWORD)a1;
        v44 = v17;
        v39 = v37;
        *((_QWORD *)v2 - 12) = v8;
        v49 = v26;
        v40 = v11;
        v20 = (__int64)v2;
        *((_QWORD *)v2 - 16) = v15;
        v42 = v14;
        v45 = v18;
        v47 = v7;
        v48 = v10;
        v41 = 1;
        *((_QWORD *)v2 - 15) = 0;
        *(v2 - 28) = 0;
        *(v2 - 27) = 0;
        *(v2 - 26) = 0;
        v46 = 1;
        *((_QWORD *)v2 - 11) = 0;
        *(v2 - 20) = 0;
        *(v2 - 19) = 0;
        v50 = v25;
        *(v2 - 18) = 0;
        v51 = 1;
        *((_QWORD *)v2 - 8) = v30;
        v52 = v29;
        *((_QWORD *)v2 - 7) = 0;
        v53 = v31;
        *(v2 - 12) = 0;
        v54 = v28;
        v57 = v35;
        *(v2 - 11) = 0;
        v55 = v27;
        *(v2 - 10) = 0;
        v21 = 0x86BCA1AF286BCA1BLL * (v19 >> 3);
        v56 = 1;
        *((_QWORD *)v2 - 4) = v36;
        *((_QWORD *)v2 - 3) = 0;
        v58 = v34;
        *(v2 - 4) = 0;
        v59 = v33;
        *(v2 - 3) = 0;
        v60 = v32;
        *(v2 - 2) = 0;
        if ( v19 > 0 )
        {
          do
          {
            v22 = v20 - 304;
            v20 -= 152;
            sub_25F6310(v20, v22);
            --v21;
          }
          while ( v21 );
        }
        v5 = (__int64)a1;
      }
      else
      {
        *((_QWORD *)v2 - 12) = v8;
        *((_QWORD *)v2 - 16) = v15;
        v49 = v26;
        v42 = v14;
        v23 = v2 - 76;
        v50 = v25;
        v39 = v37;
        *((_QWORD *)v2 - 8) = v30;
        v40 = v11;
        v52 = v29;
        v41 = 1;
        *((_QWORD *)v2 - 15) = 0;
        v43 = v16;
        *(v2 - 28) = 0;
        v44 = v17;
        *(v2 - 27) = 0;
        v45 = v18;
        *(v2 - 26) = 0;
        v46 = 1;
        v47 = v7;
        *((_QWORD *)v2 - 11) = 0;
        v48 = v10;
        *(v2 - 20) = 0;
        *(v2 - 19) = 0;
        *(v2 - 18) = 0;
        v51 = 1;
        *((_QWORD *)v2 - 7) = 0;
        v53 = v31;
        *(v2 - 12) = 0;
        v54 = v28;
        *(v2 - 11) = 0;
        v55 = v27;
        *(v2 - 10) = 0;
        *((_QWORD *)v2 - 4) = v36;
        v56 = 1;
        v57 = v35;
        *((_QWORD *)v2 - 3) = 0;
        v58 = v34;
        *(v2 - 4) = 0;
        v59 = v33;
        *(v2 - 3) = 0;
        v60 = v32;
        *(v2 - 2) = 0;
        if ( v4 < *(v2 - 76) )
        {
          do
          {
            sub_25F6310((__int64)(v23 + 38), (__int64)v23);
            v5 = (__int64)v23;
            v23 -= 38;
          }
          while ( v38[0] < *v23 );
        }
      }
      sub_25F6310(v5, (__int64)v38);
      sub_C7D6A0(v57, 8LL * v60, 4);
      sub_C7D6A0(v52, 8LL * v55, 4);
      sub_C7D6A0(v47, 16LL * v50, 8);
      sub_C7D6A0(v42, 16LL * v45, 8);
    }
  }
}
