// Function: sub_2CE18D0
// Address: 0x2ce18d0
//
__int64 __fastcall sub_2CE18D0(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4)
{
  __int64 result; // rax
  _QWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // rsi
  __int64 v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  _BYTE *v18; // rsi
  __int64 *v19; // rdx
  __int64 *v20; // r15
  __int64 v21; // r14
  _QWORD *v22; // r13
  unsigned int v23; // ecx
  __int64 v24; // r10
  __int64 v25; // rdi
  _QWORD *v26; // rdi
  __int64 *v27; // rax
  __int64 v28; // rdi
  int v29; // esi
  char v30; // dl
  int v31; // eax
  __int64 v32; // rax
  __int64 *v33; // [rsp+8h] [rbp-C8h]
  int v34; // [rsp+10h] [rbp-C0h]
  __int64 v36; // [rsp+18h] [rbp-B8h]
  unsigned int v37; // [rsp+20h] [rbp-B0h]
  unsigned int v38; // [rsp+24h] [rbp-ACh]
  __int64 v41; // [rsp+38h] [rbp-98h]
  unsigned int v42; // [rsp+44h] [rbp-8Ch] BYREF
  __int64 v43; // [rsp+48h] [rbp-88h]
  __int64 *v44; // [rsp+50h] [rbp-80h] BYREF
  _BYTE *v45; // [rsp+58h] [rbp-78h]
  _BYTE *v46; // [rsp+60h] [rbp-70h]
  _QWORD v47[4]; // [rsp+70h] [rbp-60h] BYREF
  char v48; // [rsp+90h] [rbp-40h]
  char v49; // [rsp+91h] [rbp-3Fh]

  result = *(_QWORD *)(a2 + 8);
  v36 = result;
  if ( *(_BYTE *)(result + 8) == 15 )
  {
    v42 = 0;
    result = *(unsigned int *)(result + 12);
    v34 = result;
    if ( (_DWORD)result )
    {
      v41 = a4 + 24;
      do
      {
        v49 = 1;
        v47[0] = "extract";
        v48 = 3;
        v6 = sub_BD2C40(104, 1u);
        if ( v6 )
        {
          v7 = sub_B501B0(*(_QWORD *)(a2 + 8), &v42, 1);
          sub_B44260((__int64)v6, v7, 64, 1u, v41, 0);
          if ( *(v6 - 4) )
          {
            v8 = *(v6 - 3);
            *(_QWORD *)*(v6 - 2) = v8;
            if ( v8 )
              *(_QWORD *)(v8 + 16) = *(v6 - 2);
          }
          *(v6 - 4) = a2;
          v9 = *(_QWORD *)(a2 + 16);
          *(v6 - 3) = v9;
          if ( v9 )
            *(_QWORD *)(v9 + 16) = v6 - 3;
          *(v6 - 2) = a2 + 16;
          *(_QWORD *)(a2 + 16) = v6 - 4;
          v6[9] = v6 + 11;
          v6[10] = 0x400000000LL;
          sub_B50030((__int64)v6, &v42, 1, (__int64)v47);
        }
        v44 = 0;
        v45 = 0;
        v46 = 0;
        v10 = (_QWORD *)sub_BD5C60(a4);
        v11 = sub_BCB2D0(v10);
        v12 = sub_ACD640(v11, 0, 0);
        v13 = v45;
        v47[0] = v12;
        if ( v45 == v46 )
        {
          sub_928380((__int64)&v44, v45, v47);
        }
        else
        {
          if ( v45 )
          {
            *(_QWORD *)v45 = v12;
            v13 = v45;
          }
          v45 = v13 + 8;
        }
        v14 = v42;
        v15 = (_QWORD *)sub_BD5C60(a4);
        v16 = sub_BCB2D0(v15);
        v17 = sub_ACD640(v16, v14, 0);
        v18 = v45;
        v47[0] = v17;
        if ( v45 == v46 )
        {
          sub_928380((__int64)&v44, v45, v47);
          v19 = (__int64 *)v45;
        }
        else
        {
          if ( v45 )
          {
            *(_QWORD *)v45 = v17;
            v18 = v45;
          }
          v19 = (__int64 *)(v18 + 8);
          v45 = v18 + 8;
        }
        v20 = v44;
        v49 = 1;
        v48 = 3;
        v47[0] = "gep";
        v21 = v19 - v44;
        v33 = v19;
        v22 = sub_BD2C40(88, (int)v21 + 1);
        if ( v22 )
        {
          v23 = v37 & 0xE0000000 | (v21 + 1) & 0x7FFFFFF;
          v37 = v23;
          v24 = *(_QWORD *)(a1 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 > 1 && v20 != v33 )
          {
            v27 = v20;
            v28 = *(_QWORD *)(*v20 + 8);
            v29 = *(unsigned __int8 *)(v28 + 8);
            if ( v29 == 17 )
            {
LABEL_35:
              v30 = 0;
            }
            else
            {
              while ( v29 != 18 )
              {
                if ( v33 == ++v27 )
                  goto LABEL_21;
                v28 = *(_QWORD *)(*v27 + 8);
                v29 = *(unsigned __int8 *)(v28 + 8);
                if ( v29 == 17 )
                  goto LABEL_35;
              }
              v30 = 1;
            }
            v31 = *(_DWORD *)(v28 + 32);
            BYTE4(v43) = v30;
            v38 = v23;
            LODWORD(v43) = v31;
            v32 = sub_BCE1B0((__int64 *)v24, v43);
            v23 = v38;
            v24 = v32;
          }
LABEL_21:
          sub_B44260((__int64)v22, v24, 34, v23, v41, 0);
          v22[9] = v36;
          v22[10] = sub_B4DC50(v36, (__int64)v20, v21);
          sub_B4D9A0((__int64)v22, a1, v20, v21, (__int64)v47);
        }
        sub_B4DDE0((__int64)v22, 3);
        v25 = v6[1];
        if ( *(_BYTE *)(v25 + 8) == 15 && (unsigned __int8)sub_2CDFA60(v25) )
        {
          sub_2CE18D0(v22, v6, a3, a4);
        }
        else
        {
          v26 = sub_BD2C40(80, unk_3F10A10);
          if ( v26 )
            sub_B4D3F0((__int64)v26, (__int64)v6, (__int64)v22, a3, v41, 0);
        }
        if ( v44 )
          j_j___libc_free_0((unsigned __int64)v44);
        result = v42 + 1;
        v42 = result;
      }
      while ( (_DWORD)result != v34 );
    }
  }
  return result;
}
