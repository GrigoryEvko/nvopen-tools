// Function: sub_2CE23E0
// Address: 0x2ce23e0
//
__int64 __fastcall sub_2CE23E0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v7; // r15
  __int64 v8; // rbx
  _QWORD *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  _BYTE *v12; // rsi
  __int64 v13; // r12
  _QWORD *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  _BYTE *v17; // rsi
  __int64 *v18; // rdx
  __int64 *v19; // r14
  __int64 v20; // r13
  _QWORD *v21; // r12
  unsigned int v22; // ecx
  __int64 v23; // r10
  __int64 v24; // r14
  _QWORD *v25; // rax
  __int64 v26; // r13
  __int64 v27; // rdi
  _QWORD *v28; // rax
  __int64 *v30; // rax
  __int64 v31; // rdi
  int v32; // esi
  char v33; // dl
  int v34; // eax
  __int64 v35; // rax
  _BYTE *v36; // rsi
  int v39; // [rsp+14h] [rbp-CCh]
  __int64 *v40; // [rsp+18h] [rbp-C8h]
  unsigned int v42; // [rsp+2Ch] [rbp-B4h]
  unsigned int v43; // [rsp+30h] [rbp-B0h]
  unsigned int v46; // [rsp+54h] [rbp-8Ch] BYREF
  __int64 v47; // [rsp+58h] [rbp-88h]
  __int64 *v48; // [rsp+60h] [rbp-80h] BYREF
  _BYTE *v49; // [rsp+68h] [rbp-78h]
  _BYTE *v50; // [rsp+70h] [rbp-70h]
  _QWORD v51[4]; // [rsp+80h] [rbp-60h] BYREF
  char v52; // [rsp+A0h] [rbp-40h]
  char v53; // [rsp+A1h] [rbp-3Fh]

  v5 = 0;
  if ( *(_BYTE *)(a2 + 8) == 15 )
  {
    v46 = 0;
    v7 = sub_ACA8A0((__int64 **)a2);
    v5 = v7;
    v39 = *(_DWORD *)(a2 + 12);
    if ( v39 )
    {
      v8 = a3 + 24;
      while ( 1 )
      {
        v48 = 0;
        v49 = 0;
        v50 = 0;
        v9 = (_QWORD *)sub_BD5C60(a3);
        v10 = sub_BCB2D0(v9);
        v11 = sub_ACD640(v10, 0, 0);
        v12 = v49;
        v51[0] = v11;
        if ( v49 == v50 )
        {
          sub_928380((__int64)&v48, v49, v51);
        }
        else
        {
          if ( v49 )
          {
            *(_QWORD *)v49 = v11;
            v12 = v49;
          }
          v49 = v12 + 8;
        }
        v13 = v46;
        v14 = (_QWORD *)sub_BD5C60(a3);
        v15 = sub_BCB2D0(v14);
        v16 = sub_ACD640(v15, v13, 0);
        v17 = v49;
        v51[0] = v16;
        if ( v49 == v50 )
        {
          sub_928380((__int64)&v48, v49, v51);
          v18 = (__int64 *)v49;
        }
        else
        {
          if ( v49 )
          {
            *(_QWORD *)v49 = v16;
            v17 = v49;
          }
          v18 = (__int64 *)(v17 + 8);
          v49 = v17 + 8;
        }
        v19 = v48;
        v53 = 1;
        v52 = 3;
        v51[0] = "gep";
        v20 = v18 - v48;
        v40 = v18;
        v21 = sub_BD2C40(88, (int)v20 + 1);
        if ( v21 )
        {
          v22 = v42 & 0xE0000000 | (v20 + 1) & 0x7FFFFFF;
          v42 = v22;
          v23 = *(_QWORD *)(a1 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v23 + 8) - 17 > 1 && v19 != v40 )
          {
            v30 = v19;
            v31 = *(_QWORD *)(*v19 + 8);
            v32 = *(unsigned __int8 *)(v31 + 8);
            if ( v32 == 17 )
            {
LABEL_31:
              v33 = 0;
            }
            else
            {
              while ( v32 != 18 )
              {
                if ( v40 == ++v30 )
                  goto LABEL_14;
                v31 = *(_QWORD *)(*v30 + 8);
                v32 = *(unsigned __int8 *)(v31 + 8);
                if ( v32 == 17 )
                  goto LABEL_31;
              }
              v33 = 1;
            }
            v34 = *(_DWORD *)(v31 + 32);
            BYTE4(v47) = v33;
            v43 = v22;
            LODWORD(v47) = v34;
            v35 = sub_BCE1B0((__int64 *)v23, v47);
            v22 = v43;
            v23 = v35;
          }
LABEL_14:
          sub_B44260((__int64)v21, v23, 34, v22, v8, 0);
          v21[9] = a2;
          v21[10] = sub_B4DC50(a2, (__int64)v19, v20);
          sub_B4D9A0((__int64)v21, a1, v19, v20, (__int64)v51);
        }
        sub_B4DDE0((__int64)v21, 3);
        v24 = v21[10];
        v51[0] = "loadfield";
        v53 = 1;
        v52 = 3;
        v25 = sub_BD2C40(80, 1u);
        v26 = (__int64)v25;
        if ( v25 )
          sub_B4D1B0((__int64)v25, v24, (__int64)v21, (__int64)v51, a4, 0, v8, 0);
        v27 = *(_QWORD *)(v26 + 8);
        if ( *(_BYTE *)(v27 + 8) == 15 && (unsigned __int8)sub_2CDFA60(v27) )
        {
          v51[0] = v26;
          v36 = *(_BYTE **)(a5 + 8);
          if ( v36 == *(_BYTE **)(a5 + 16) )
          {
            sub_249A840(a5, v36, v51);
          }
          else
          {
            if ( v36 )
            {
              *(_QWORD *)v36 = v26;
              v36 = *(_BYTE **)(a5 + 8);
            }
            *(_QWORD *)(a5 + 8) = v36 + 8;
          }
          v26 = sub_2CE23E0(v21, *(_QWORD *)(v26 + 8), a3, a4, a5);
        }
        v53 = 1;
        v51[0] = "insertfield";
        v52 = 3;
        v28 = sub_BD2C40(104, unk_3F148BC);
        v5 = (__int64)v28;
        if ( v28 )
        {
          sub_B44260((__int64)v28, *(_QWORD *)(v7 + 8), 65, 2u, v8, 0);
          *(_QWORD *)(v5 + 72) = v5 + 88;
          *(_QWORD *)(v5 + 80) = 0x400000000LL;
          sub_B4FD20(v5, v7, v26, &v46, 1, (__int64)v51);
        }
        if ( v48 )
          j_j___libc_free_0((unsigned __int64)v48);
        if ( v39 == ++v46 )
          break;
        v7 = v5;
      }
    }
  }
  return v5;
}
