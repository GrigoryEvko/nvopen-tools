// Function: sub_E79920
// Address: 0xe79920
//
__int64 __fastcall sub_E79920(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rdi
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  char v17; // r15
  char v18; // r11
  char v19; // r10
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rcx
  __int64 v23; // rcx
  unsigned __int64 v24; // rax
  _QWORD *v25; // r9
  int v26; // [rsp-6Ch] [rbp-6Ch]
  int v27; // [rsp-68h] [rbp-68h]
  unsigned int v28; // [rsp-64h] [rbp-64h]
  __int64 v29; // [rsp-60h] [rbp-60h]
  __int64 v30; // [rsp-58h] [rbp-58h]
  __int64 v31; // [rsp-50h] [rbp-50h]
  __int64 v32; // [rsp-48h] [rbp-48h]
  int v33; // [rsp-40h] [rbp-40h]
  char v34; // [rsp-39h] [rbp-39h]

  result = 0x155555555555555LL;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 <= 0x155555555555555LL )
    result = a3;
  if ( a3 > 0 )
  {
    v4 = result;
    while ( 1 )
    {
      v7 = 96 * v4;
      result = sub_2207800(96 * v4, &unk_435FF63);
      v8 = result;
      if ( result )
        break;
      v4 >>= 1;
      if ( !v4 )
        return result;
    }
    v9 = *a2;
    v10 = a2[1];
    v11 = result + v7;
    v12 = a2[2];
    v13 = a2[3];
    *(_QWORD *)result = *a2;
    *(_QWORD *)(result + 8) = v10;
    *(_QWORD *)(result + 16) = v12;
    *(_QWORD *)(result + 24) = v13;
    v14 = a2[4];
    a2[4] = 0;
    v31 = v14;
    *(_QWORD *)(v8 + 32) = v14;
    v15 = a2[5];
    a2[5] = 0;
    v30 = v15;
    *(_QWORD *)(v8 + 40) = v15;
    v16 = a2[6];
    a2[6] = 0;
    v29 = v16;
    *(_QWORD *)(v8 + 48) = v16;
    v26 = *((_DWORD *)a2 + 14);
    *(_DWORD *)(v8 + 56) = v26;
    v33 = *((_DWORD *)a2 + 15);
    *(_DWORD *)(v8 + 60) = v33;
    v27 = *((_DWORD *)a2 + 16);
    *(_DWORD *)(v8 + 64) = v27;
    v32 = a2[9];
    *(_QWORD *)(v8 + 72) = v32;
    v17 = *((_BYTE *)a2 + 81);
    v18 = *((_BYTE *)a2 + 88);
    v34 = *((_BYTE *)a2 + 80);
    v19 = *((_BYTE *)a2 + 89);
    *(_BYTE *)(v8 + 80) = v34;
    LODWORD(v16) = *((_DWORD *)a2 + 21);
    *(_BYTE *)(v8 + 81) = v17;
    v28 = v16;
    *(_DWORD *)(v8 + 84) = v16;
    v20 = v8 + 96;
    *(_BYTE *)(v8 + 88) = v18;
    *(_BYTE *)(v8 + 89) = v19;
    if ( v11 == v8 + 96 )
    {
      v25 = (_QWORD *)v8;
    }
    else
    {
      while ( 1 )
      {
        *(_QWORD *)(v20 + 24) = v13;
        v21 = *(_QWORD *)(v20 - 64);
        v20 += 96;
        *(_QWORD *)(v20 - 96) = v9;
        *(_QWORD *)(v20 - 64) = v21;
        v22 = *(_QWORD *)(v20 - 152);
        *(_QWORD *)(v20 - 88) = v10;
        *(_QWORD *)(v20 - 56) = v22;
        v23 = *(_QWORD *)(v20 - 144);
        *(_QWORD *)(v20 - 80) = v12;
        *(_QWORD *)(v20 - 48) = v23;
        LODWORD(v23) = *(_DWORD *)(v20 - 136);
        *(_QWORD *)(v20 - 144) = 0;
        *(_DWORD *)(v20 - 40) = v23;
        LODWORD(v23) = *(_DWORD *)(v20 - 132);
        *(_QWORD *)(v20 - 152) = 0;
        *(_DWORD *)(v20 - 36) = v23;
        LODWORD(v23) = *(_DWORD *)(v20 - 128);
        *(_QWORD *)(v20 - 160) = 0;
        *(_DWORD *)(v20 - 32) = v23;
        *(_QWORD *)(v20 - 24) = *(_QWORD *)(v20 - 120);
        *(_BYTE *)(v20 - 16) = *(_BYTE *)(v20 - 112);
        *(_BYTE *)(v20 - 15) = *(_BYTE *)(v20 - 111);
        *(_DWORD *)(v20 - 12) = *(_DWORD *)(v20 - 108);
        *(_BYTE *)(v20 - 8) = *(_BYTE *)(v20 - 104);
        *(_BYTE *)(v20 - 7) = *(_BYTE *)(v20 - 103);
        if ( v11 == v20 )
          break;
        v9 = *(_QWORD *)(v20 - 96);
        v10 = *(_QWORD *)(v20 - 88);
        v12 = *(_QWORD *)(v20 - 80);
        v13 = *(_QWORD *)(v20 - 72);
      }
      v24 = v8 + 96 * ((0x2AAAAAAAAAAAAABLL * ((unsigned __int64)(v7 - 192) >> 5)) & 0x7FFFFFFFFFFFFFFLL);
      v25 = (_QWORD *)(v8
                     + 32
                     * (3 * ((0x2AAAAAAAAAAAAABLL * ((unsigned __int64)(v7 - 192) >> 5)) & 0x7FFFFFFFFFFFFFFLL) + 3));
      v9 = *(_QWORD *)(v24 + 96);
      v10 = *(_QWORD *)(v24 + 104);
      v26 = *(_DWORD *)(v24 + 152);
      v12 = *(_QWORD *)(v24 + 112);
      v33 = *(_DWORD *)(v24 + 156);
      v13 = *(_QWORD *)(v24 + 120);
      v27 = *(_DWORD *)(v24 + 160);
      v32 = *(_QWORD *)(v24 + 168);
      v34 = *(_BYTE *)(v24 + 176);
      v17 = *(_BYTE *)(v24 + 177);
      v28 = *(_DWORD *)(v24 + 180);
      v18 = *(_BYTE *)(v24 + 184);
      v31 = *(_QWORD *)(v24 + 128);
      v19 = *(_BYTE *)(v24 + 185);
      v30 = *(_QWORD *)(v24 + 136);
      v29 = *(_QWORD *)(v24 + 144);
    }
    *a2 = v9;
    a2[1] = v10;
    a2[4] = v31;
    a2[2] = v12;
    a2[5] = v30;
    a2[3] = v13;
    a2[6] = v29;
    v25[4] = 0;
    *((_DWORD *)a2 + 14) = v26;
    v25[5] = 0;
    *((_DWORD *)a2 + 15) = v33;
    v25[6] = 0;
    *((_DWORD *)a2 + 16) = v27;
    *((_BYTE *)a2 + 81) = v17;
    a2[9] = v32;
    *((_BYTE *)a2 + 88) = v18;
    *((_BYTE *)a2 + 80) = v34;
    *((_BYTE *)a2 + 89) = v19;
    *((_DWORD *)a2 + 21) = v28;
    a1[2] = v8;
    a1[1] = v4;
    return v28;
  }
  return result;
}
