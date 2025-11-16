// Function: sub_24E9B00
// Address: 0x24e9b00
//
void __fastcall sub_24E9B00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 *v4; // rax
  unsigned __int8 *v5; // r13
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r12
  int v13; // r12d
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rbx
  __int64 v17; // rax
  char v18; // al
  char v19; // r15
  _QWORD *v20; // rax
  __int64 v21; // r9
  __int64 v22; // r13
  __int64 v23; // r15
  _BYTE *v24; // rbx
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 *v27; // r13
  __int64 v28; // rax
  char v29; // al
  char v30; // r15
  _QWORD *v31; // rax
  __int64 v32; // rbx
  __int64 v33; // r13
  _BYTE *v34; // r12
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // r12
  __int64 v38; // rdx
  __int64 v39; // r13
  __int64 v40; // r15
  __int64 v41; // r12
  __int64 v42; // rdx
  __int64 v43; // rbx
  __int64 v44; // r15
  __int64 v45; // rcx
  __int64 v46; // rcx
  __int64 *v47; // [rsp+0h] [rbp-190h]
  unsigned __int8 **v49; // [rsp+10h] [rbp-180h]
  __int64 *v51; // [rsp+48h] [rbp-148h]
  __int64 v52; // [rsp+58h] [rbp-138h] BYREF
  __int64 *v53; // [rsp+60h] [rbp-130h] BYREF
  __int64 v54; // [rsp+68h] [rbp-128h]
  char v55[32]; // [rsp+70h] [rbp-120h] BYREF
  __int16 v56; // [rsp+90h] [rbp-100h]
  _BYTE v57[32]; // [rsp+A0h] [rbp-F0h] BYREF
  __int16 v58; // [rsp+C0h] [rbp-D0h]
  _BYTE *v59; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v60; // [rsp+D8h] [rbp-B8h]
  _BYTE v61[32]; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v62; // [rsp+100h] [rbp-90h]
  __int64 v63; // [rsp+108h] [rbp-88h]
  __int64 v64; // [rsp+110h] [rbp-80h]
  __int64 v65; // [rsp+118h] [rbp-78h]
  void **v66; // [rsp+120h] [rbp-70h]
  void **v67; // [rsp+128h] [rbp-68h]
  __int64 v68; // [rsp+130h] [rbp-60h]
  int v69; // [rsp+138h] [rbp-58h]
  __int16 v70; // [rsp+13Ch] [rbp-54h]
  char v71; // [rsp+13Eh] [rbp-52h]
  __int64 v72; // [rsp+140h] [rbp-50h]
  __int64 v73; // [rsp+148h] [rbp-48h]
  void *v74; // [rsp+150h] [rbp-40h] BYREF
  void *v75; // [rsp+158h] [rbp-38h] BYREF

  if ( *(_DWORD *)(a2 + 280) != 3 || *(_DWORD *)(a2 + 128) )
  {
    v52 = 0;
    v53 = &v52;
    v3 = *(unsigned int *)(a2 + 256);
    v4 = *(__int64 **)(a2 + 248);
    v54 = a1;
    v49 = (unsigned __int8 **)&v4[v3];
    if ( v49 != (unsigned __int8 **)v4 )
    {
      v51 = v4;
      do
      {
        v5 = (unsigned __int8 *)*v51;
        v6 = *v51;
        if ( a3 )
        {
          v59 = (_BYTE *)*v51;
          v6 = sub_24E84F0(a3, (__int64 *)&v59)[2];
        }
        v7 = sub_BD5C60(v6);
        v71 = 7;
        v65 = v7;
        v68 = 0;
        v66 = &v74;
        v59 = v61;
        v67 = &v75;
        v60 = 0x200000000LL;
        v69 = 0;
        v74 = &unk_49DA100;
        v70 = 512;
        LOWORD(v64) = 0;
        v75 = &unk_49DA0B0;
        v72 = 0;
        v73 = 0;
        v62 = 0;
        v63 = 0;
        sub_D5F1F0((__int64)&v59, v6);
        v9 = *v5;
        if ( (_DWORD)v9 == 40 )
        {
          v10 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v5);
        }
        else
        {
          v10 = -32;
          if ( (_DWORD)v9 != 85 )
          {
            if ( (_DWORD)v9 != 34 )
              BUG();
            v10 = -96;
          }
        }
        if ( (v5[7] & 0x80u) != 0 )
        {
          v11 = sub_BD2BC0((__int64)v5);
          v12 = v11 + v9;
          if ( (v5[7] & 0x80u) == 0 )
          {
            if ( (unsigned int)(v12 >> 4) )
LABEL_61:
              BUG();
          }
          else if ( (unsigned int)((v12 - sub_BD2BC0((__int64)v5)) >> 4) )
          {
            if ( (v5[7] & 0x80u) == 0 )
              goto LABEL_61;
            v13 = *(_DWORD *)(sub_BD2BC0((__int64)v5) + 8);
            if ( (v5[7] & 0x80u) == 0 )
              BUG();
            v14 = sub_BD2BC0((__int64)v5);
            v10 -= 32LL * (unsigned int)(*(_DWORD *)(v14 + v9 - 4) - v13);
          }
        }
        v15 = *v53;
        if ( 32LL * (*((_DWORD *)v5 + 1) & 0x7FFFFFF) + v10 )
        {
          v16 = *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
          if ( !v15 )
          {
            v37 = v54;
            v47 = *(__int64 **)(v16 + 8);
            if ( (*(_BYTE *)(v54 + 2) & 1) != 0 )
            {
              sub_B2C6D0(v54, v6, v9, v8);
              v38 = *(_QWORD *)(v37 + 96);
              v39 = v38 + 40LL * *(_QWORD *)(v37 + 104);
              if ( (*(_BYTE *)(v37 + 2) & 1) != 0 )
              {
                sub_B2C6D0(v37, v6, v38, v46);
                v38 = *(_QWORD *)(v37 + 96);
              }
            }
            else
            {
              v38 = *(_QWORD *)(v54 + 96);
              v39 = v38 + 40LL * *(_QWORD *)(v54 + 104);
            }
            v40 = v38;
            if ( v39 == v38 )
            {
LABEL_43:
              v15 = sub_24E7AB0((__int64)&v53, v47);
            }
            else
            {
              while ( 1 )
              {
                v15 = v40;
                if ( (unsigned __int8)sub_BD6020(v40) )
                  break;
                v40 += 40;
                if ( v39 == v40 )
                  goto LABEL_43;
              }
              *v53 = v40;
            }
          }
          v17 = sub_AA4E30(v62);
          v18 = sub_AE5020(v17, *(_QWORD *)(v16 + 8));
          v58 = 257;
          v19 = v18;
          v20 = sub_BD2C40(80, unk_3F10A10);
          v22 = (__int64)v20;
          if ( v20 )
            sub_B4D3C0((__int64)v20, v16, v15, 0, v19, v21, 0, 0);
          (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v67 + 2))(v67, v22, v57, v63, v64);
          v23 = (__int64)v59;
          v24 = &v59[16 * (unsigned int)v60];
          if ( v59 != v24 )
          {
            do
            {
              v25 = *(_QWORD *)(v23 + 8);
              v26 = *(_DWORD *)v23;
              v23 += 16;
              sub_B99FD0(v22, v26, v25);
            }
            while ( v24 != (_BYTE *)v23 );
          }
        }
        else
        {
          v27 = (__int64 *)*((_QWORD *)v5 + 1);
          if ( !v15 )
          {
            v41 = v54;
            if ( (*(_BYTE *)(v54 + 2) & 1) != 0 )
            {
              sub_B2C6D0(v54, v6, v9, v8);
              v42 = *(_QWORD *)(v41 + 96);
              v43 = v42 + 40LL * *(_QWORD *)(v41 + 104);
              if ( (*(_BYTE *)(v41 + 2) & 1) != 0 )
              {
                sub_B2C6D0(v41, v6, v42, v45);
                v42 = *(_QWORD *)(v41 + 96);
              }
            }
            else
            {
              v42 = *(_QWORD *)(v54 + 96);
              v43 = v42 + 40LL * *(_QWORD *)(v54 + 104);
            }
            v44 = v42;
            if ( v42 == v43 )
            {
LABEL_55:
              v15 = sub_24E7AB0((__int64)&v53, v27);
            }
            else
            {
              while ( 1 )
              {
                v15 = v44;
                if ( (unsigned __int8)sub_BD6020(v44) )
                  break;
                v44 += 40;
                if ( v43 == v44 )
                  goto LABEL_55;
              }
              *v53 = v44;
            }
          }
          v56 = 257;
          v28 = sub_AA4E30(v62);
          v29 = sub_AE5020(v28, (__int64)v27);
          v58 = 257;
          v30 = v29;
          v31 = sub_BD2C40(80, unk_3F10A14);
          v32 = (__int64)v31;
          if ( v31 )
            sub_B4D190((__int64)v31, (__int64)v27, v15, (__int64)v57, 0, v30, 0, 0);
          (*((void (__fastcall **)(void **, __int64, char *, __int64, __int64))*v67 + 2))(v67, v32, v55, v63, v64);
          v33 = (__int64)v59;
          v34 = &v59[16 * (unsigned int)v60];
          if ( v59 != v34 )
          {
            do
            {
              v35 = *(_QWORD *)(v33 + 8);
              v36 = *(_DWORD *)v33;
              v33 += 16;
              sub_B99FD0(v32, v36, v35);
            }
            while ( v34 != (_BYTE *)v33 );
          }
          v15 = v32;
        }
        sub_BD84D0(v6, v15);
        sub_B43D60((_QWORD *)v6);
        nullsub_61();
        v74 = &unk_49DA100;
        nullsub_63();
        if ( v59 != v61 )
          _libc_free((unsigned __int64)v59);
        ++v51;
      }
      while ( v49 != (unsigned __int8 **)v51 );
    }
    if ( !a3 )
      *(_DWORD *)(a2 + 256) = 0;
  }
}
