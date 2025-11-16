// Function: sub_2222A70
// Address: 0x2222a70
//
__int64 __fastcall sub_2222A70(__int64 a1, signed __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // r13

  v2 = sub_2252480(a1, &`typeinfo for'std::locale::facet, &`typeinfo for'std::locale::facet::__shim, -2);
  if ( v2 )
    return *(_QWORD *)v2;
  if ( a2 == &qword_4FD6868 )
  {
    v5 = sub_22077B0(0x90u);
    *(_DWORD *)(v5 + 8) = 0;
    v6 = v5;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 24) = 0;
    *(_QWORD *)v5 = off_4A04910;
    *(_BYTE *)(v5 + 32) = 0;
    *(_QWORD *)(v5 + 40) = 0;
    *(_QWORD *)(v5 + 48) = 0;
    *(_QWORD *)(v5 + 56) = 0;
    *(_QWORD *)(v5 + 64) = 0;
    *(_WORD *)(v5 + 72) = 0;
    *(_BYTE *)(v5 + 136) = 0;
    v3 = sub_22077B0(0x28u);
    *(_DWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = v6;
    *(_QWORD *)v3 = off_4A05678;
    sub_220E100(v3, 0);
    *(_QWORD *)(v3 + 24) = a1;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)(v3 + 32) = v6;
    *(_QWORD *)v3 = off_4A05C00;
    sub_2212A60((__int64 *)a1, v6);
  }
  else if ( a2 == &qword_4FD6850 )
  {
    v9 = sub_22077B0(0x20u);
    *(_DWORD *)(v9 + 8) = 0;
    v3 = v9;
    *(_QWORD *)v9 = off_4A05640;
    *(_QWORD *)(v9 + 16) = sub_2208E60(32, &`typeinfo for'std::locale::facet);
    *(_QWORD *)(v3 + 24) = a1;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)v3 = off_4A05C48;
  }
  else if ( a2 == &qword_4FD6860 )
  {
    v7 = sub_22077B0(0x18u);
    *(_DWORD *)(v7 + 8) = 0;
    v3 = v7;
    *(_QWORD *)(v7 + 16) = a1;
    *(_QWORD *)v7 = off_4A05828;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)v7 = off_4A05EA0;
  }
  else if ( a2 == &qword_4FD6878 )
  {
    v10 = sub_22077B0(0x18u);
    *(_DWORD *)(v10 + 8) = 0;
    v3 = v10;
    *(_QWORD *)(v10 + 16) = a1;
    *(_QWORD *)v10 = off_4A057C8;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)v10 = off_49D28B8;
  }
  else if ( a2 == &qword_4FD6870 )
  {
    v11 = sub_22077B0(0x18u);
    *(_DWORD *)(v11 + 8) = 0;
    v3 = v11;
    *(_QWORD *)(v11 + 16) = a1;
    *(_QWORD *)v11 = off_4A057F8;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)v11 = off_49D28E8;
  }
  else if ( a2 == &qword_4FD6880 )
  {
    v12 = sub_22077B0(0x70u);
    *(_DWORD *)(v12 + 8) = 0;
    v13 = v12;
    *(_WORD *)(v12 + 32) = 0;
    *(_QWORD *)(v12 + 16) = 0;
    *(_QWORD *)v12 = off_4A04860;
    *(_QWORD *)(v12 + 24) = 0;
    *(_BYTE *)(v12 + 34) = 0;
    *(_QWORD *)(v12 + 40) = 0;
    *(_QWORD *)(v12 + 48) = 0;
    *(_QWORD *)(v12 + 56) = 0;
    *(_QWORD *)(v12 + 64) = 0;
    *(_QWORD *)(v12 + 72) = 0;
    *(_QWORD *)(v12 + 80) = 0;
    *(_QWORD *)(v12 + 88) = 0;
    *(_DWORD *)(v12 + 96) = 0;
    *(_BYTE *)(v12 + 111) = 0;
    v3 = sub_22077B0(0x28u);
    *(_DWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = v13;
    *(_QWORD *)v3 = off_4A056C0;
    sub_220AA70(v3, 0);
    *(_QWORD *)(v3 + 24) = a1;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)(v3 + 32) = v13;
    *(_QWORD *)v3 = off_4A05C80;
    sub_22130C0((__int64 *)a1, v13);
  }
  else if ( a2 == &qword_4FD6888 )
  {
    v14 = sub_22077B0(0x70u);
    *(_DWORD *)(v14 + 8) = 0;
    v15 = v14;
    *(_QWORD *)(v14 + 16) = 0;
    *(_QWORD *)(v14 + 24) = 0;
    *(_QWORD *)v14 = off_4A04880;
    *(_WORD *)(v14 + 32) = 0;
    *(_BYTE *)(v14 + 34) = 0;
    *(_QWORD *)(v14 + 40) = 0;
    *(_QWORD *)(v14 + 48) = 0;
    *(_QWORD *)(v14 + 56) = 0;
    *(_QWORD *)(v14 + 64) = 0;
    *(_QWORD *)(v14 + 72) = 0;
    *(_QWORD *)(v14 + 80) = 0;
    *(_QWORD *)(v14 + 88) = 0;
    *(_DWORD *)(v14 + 96) = 0;
    *(_BYTE *)(v14 + 111) = 0;
    v3 = sub_22077B0(0x28u);
    *(_DWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = v15;
    *(_QWORD *)v3 = off_4A05728;
    sub_220AF80(v3, 0);
    *(_QWORD *)(v3 + 24) = a1;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)(v3 + 32) = v15;
    *(_QWORD *)v3 = off_4A05CE8;
    sub_22133C0((__int64 *)a1, v15);
  }
  else if ( a2 == &qword_4FD6858 )
  {
    v3 = sub_22077B0(0x28u);
    sub_221F830(v3, 0);
    *(_QWORD *)(v3 + 32) = a1;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)v3 = off_49D2918;
  }
  else if ( a2 == &qword_4FD68E8 )
  {
    v16 = sub_22077B0(0x150u);
    *(_DWORD *)(v16 + 8) = 0;
    v17 = v16;
    *(_QWORD *)(v16 + 16) = 0;
    *(_QWORD *)(v16 + 24) = 0;
    *(_QWORD *)v16 = off_4A04930;
    *(_BYTE *)(v16 + 32) = 0;
    *(_QWORD *)(v16 + 40) = 0;
    *(_QWORD *)(v16 + 48) = 0;
    *(_QWORD *)(v16 + 56) = 0;
    *(_QWORD *)(v16 + 64) = 0;
    *(_QWORD *)(v16 + 72) = 0;
    *(_BYTE *)(v16 + 328) = 0;
    v3 = sub_22077B0(0x28u);
    *(_DWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = v17;
    *(_QWORD *)v3 = off_4A060C8;
    sub_220E3D0(v3, 0);
    *(_QWORD *)(v3 + 24) = a1;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)(v3 + 32) = v17;
    *(_QWORD *)v3 = off_4A05D50;
    sub_2212CA0((__int64 *)a1, v17);
  }
  else if ( a2 == &qword_4FD68D0 )
  {
    v8 = sub_22077B0(0x20u);
    *(_DWORD *)(v8 + 8) = 0;
    v3 = v8;
    *(_QWORD *)v8 = off_4A06090;
    *(_QWORD *)(v8 + 16) = sub_2208E60(32, &`typeinfo for'std::locale::facet);
    *(_QWORD *)(v3 + 24) = a1;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)v3 = off_4A05D98;
  }
  else if ( a2 == &qword_4FD68E0 )
  {
    v18 = sub_22077B0(0x18u);
    *(_DWORD *)(v18 + 8) = 0;
    v3 = v18;
    *(_QWORD *)(v18 + 16) = a1;
    *(_QWORD *)v18 = off_4A06278;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)v18 = off_4A05EF8;
  }
  else if ( a2 == &qword_4FD68F8 )
  {
    v19 = sub_22077B0(0x18u);
    *(_DWORD *)(v19 + 8) = 0;
    v3 = v19;
    *(_QWORD *)(v19 + 16) = a1;
    *(_QWORD *)v19 = off_4A06218;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)v19 = off_49D2950;
  }
  else if ( a2 == &qword_4FD68F0 )
  {
    v20 = sub_22077B0(0x18u);
    *(_DWORD *)(v20 + 8) = 0;
    v3 = v20;
    *(_QWORD *)(v20 + 16) = a1;
    *(_QWORD *)v20 = off_4A06248;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)v20 = off_49D2980;
  }
  else if ( a2 == &qword_4FD6900 )
  {
    v21 = sub_22077B0(0xA0u);
    *(_DWORD *)(v21 + 8) = 0;
    v22 = v21;
    *(_QWORD *)(v21 + 16) = 0;
    *(_QWORD *)(v21 + 24) = 0;
    *(_QWORD *)v21 = off_4A048A0;
    *(_BYTE *)(v21 + 32) = 0;
    *(_QWORD *)(v21 + 36) = 0;
    *(_QWORD *)(v21 + 48) = 0;
    *(_QWORD *)(v21 + 56) = 0;
    *(_QWORD *)(v21 + 64) = 0;
    *(_QWORD *)(v21 + 72) = 0;
    *(_QWORD *)(v21 + 80) = 0;
    *(_QWORD *)(v21 + 88) = 0;
    *(_QWORD *)(v21 + 96) = 0;
    *(_DWORD *)(v21 + 104) = 0;
    *(_BYTE *)(v21 + 152) = 0;
    v3 = sub_22077B0(0x28u);
    *(_DWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = v22;
    *(_QWORD *)v3 = off_4A06110;
    sub_220B660(v3, 0);
    *(_QWORD *)(v3 + 24) = a1;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)(v3 + 32) = v22;
    *(_QWORD *)v3 = off_4A05DD0;
    sub_22136C0((__int64 *)a1, v22);
  }
  else if ( a2 == &qword_4FD6908 )
  {
    v23 = sub_22077B0(0xA0u);
    *(_DWORD *)(v23 + 8) = 0;
    v24 = v23;
    *(_QWORD *)(v23 + 16) = 0;
    *(_QWORD *)(v23 + 24) = 0;
    *(_QWORD *)v23 = off_4A048C0;
    *(_BYTE *)(v23 + 32) = 0;
    *(_QWORD *)(v23 + 36) = 0;
    *(_QWORD *)(v23 + 48) = 0;
    *(_QWORD *)(v23 + 56) = 0;
    *(_QWORD *)(v23 + 64) = 0;
    *(_QWORD *)(v23 + 72) = 0;
    *(_QWORD *)(v23 + 80) = 0;
    *(_QWORD *)(v23 + 88) = 0;
    *(_QWORD *)(v23 + 96) = 0;
    *(_DWORD *)(v23 + 104) = 0;
    *(_BYTE *)(v23 + 152) = 0;
    v3 = sub_22077B0(0x28u);
    *(_DWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = v24;
    *(_QWORD *)v3 = off_4A06178;
    sub_220BBD0(v3, 0);
    *(_QWORD *)(v3 + 24) = a1;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)(v3 + 32) = v24;
    *(_QWORD *)v3 = off_4A05E38;
    sub_2213A40((__int64 *)a1, v24);
  }
  else
  {
    if ( a2 != &qword_4FD68D8 )
      sub_426248((__int64)"cannot create shim for unknown locale::facet");
    v3 = sub_22077B0(0x28u);
    sub_222AA00(v3, 0);
    *(_QWORD *)(v3 + 32) = a1;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
    else
      ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)v3 = off_49D29B0;
  }
  return v3;
}
