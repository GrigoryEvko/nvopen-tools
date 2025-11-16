// Function: sub_25FEA90
// Address: 0x25fea90
//
void __fastcall sub_25FEA90(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rax
  int v9; // edx
  int v10; // edx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // [rsp+8h] [rbp-38h]

  v3 = 0xD79435E50D7943LL;
  *a1 = a3;
  a1[1] = 0;
  if ( a3 <= 0xD79435E50D7943LL )
    v3 = a3;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    while ( 1 )
    {
      v5 = 152 * v3;
      v6 = sub_2207800(152 * v3);
      v7 = v6;
      if ( v6 )
        break;
      v3 >>= 1;
      if ( !v3 )
        return;
    }
    v17 = v6 + v5;
    sub_25FE910(v6, a2);
    v8 = v7 + 152;
    if ( v17 == v7 + 152 )
    {
      v16 = v7;
    }
    else
    {
      do
      {
        v9 = *(_DWORD *)(v8 - 152);
        ++*(_QWORD *)(v8 - 128);
        ++*(_QWORD *)(v8 - 96);
        *(_DWORD *)v8 = v9;
        v10 = *(_DWORD *)(v8 - 148);
        *(_QWORD *)(v8 + 24) = 1;
        *(_DWORD *)(v8 + 4) = v10;
        v11 = *(_QWORD *)(v8 - 144);
        *(_QWORD *)(v8 + 56) = 1;
        *(_QWORD *)(v8 + 8) = v11;
        *(_QWORD *)(v8 + 16) = *(_QWORD *)(v8 - 136);
        v12 = *(_QWORD *)(v8 - 120);
        *(_QWORD *)(v8 - 120) = 0;
        *(_QWORD *)(v8 + 32) = v12;
        LODWORD(v12) = *(_DWORD *)(v8 - 112);
        *(_DWORD *)(v8 - 112) = 0;
        *(_DWORD *)(v8 + 40) = v12;
        LODWORD(v12) = *(_DWORD *)(v8 - 108);
        *(_DWORD *)(v8 - 108) = 0;
        *(_DWORD *)(v8 + 44) = v12;
        LODWORD(v12) = *(_DWORD *)(v8 - 104);
        *(_DWORD *)(v8 - 104) = 0;
        *(_DWORD *)(v8 + 48) = v12;
        v13 = *(_QWORD *)(v8 - 88);
        *(_QWORD *)(v8 - 88) = 0;
        *(_QWORD *)(v8 + 64) = v13;
        LODWORD(v13) = *(_DWORD *)(v8 - 80);
        *(_DWORD *)(v8 - 80) = 0;
        *(_DWORD *)(v8 + 72) = v13;
        LODWORD(v13) = *(_DWORD *)(v8 - 76);
        ++*(_QWORD *)(v8 - 64);
        *(_DWORD *)(v8 + 76) = v13;
        LODWORD(v13) = *(_DWORD *)(v8 - 72);
        ++*(_QWORD *)(v8 - 32);
        v8 += 152;
        *(_DWORD *)(v8 - 72) = v13;
        v14 = *(_QWORD *)(v8 - 208);
        *(_DWORD *)(v8 - 228) = 0;
        *(_QWORD *)(v8 - 56) = v14;
        LODWORD(v14) = *(_DWORD *)(v8 - 200);
        *(_DWORD *)(v8 - 224) = 0;
        *(_DWORD *)(v8 - 48) = v14;
        LODWORD(v14) = *(_DWORD *)(v8 - 196);
        *(_QWORD *)(v8 - 64) = 1;
        *(_DWORD *)(v8 - 44) = v14;
        LODWORD(v14) = *(_DWORD *)(v8 - 192);
        *(_QWORD *)(v8 - 208) = 0;
        *(_DWORD *)(v8 - 40) = v14;
        v15 = *(_QWORD *)(v8 - 176);
        *(_DWORD *)(v8 - 200) = 0;
        *(_QWORD *)(v8 - 24) = v15;
        LODWORD(v15) = *(_DWORD *)(v8 - 168);
        *(_DWORD *)(v8 - 196) = 0;
        *(_DWORD *)(v8 - 16) = v15;
        LODWORD(v15) = *(_DWORD *)(v8 - 164);
        *(_DWORD *)(v8 - 192) = 0;
        *(_DWORD *)(v8 - 12) = v15;
        *(_QWORD *)(v8 - 32) = 1;
        *(_QWORD *)(v8 - 176) = 0;
        *(_DWORD *)(v8 - 168) = 0;
        *(_DWORD *)(v8 - 164) = 0;
        LODWORD(v15) = *(_DWORD *)(v8 - 160);
        *(_DWORD *)(v8 - 160) = 0;
        *(_DWORD *)(v8 - 8) = v15;
      }
      while ( v17 != v8 );
      v16 = v7 + 152 * (((0x6BCA1AF286BCA1BLL * ((unsigned __int64)(v5 - 304) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 1);
    }
    sub_25F6310((__int64)a2, v16);
    a1[2] = v7;
    a1[1] = v3;
  }
}
