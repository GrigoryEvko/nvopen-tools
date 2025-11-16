// Function: sub_33E5830
// Address: 0x33e5830
//
__int64 __fastcall sub_33E5830(_QWORD *a1, unsigned __int16 *a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // r12
  unsigned __int16 *v7; // r8
  __int64 v8; // rax
  unsigned __int64 v9; // r15
  __int64 v10; // r9
  unsigned __int64 v11; // r15
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 *v14; // r8
  _QWORD *v15; // r15
  __int64 v16; // r13
  __int64 v18; // rsi
  __int64 v19; // rdx
  unsigned __int64 *v20; // r15
  unsigned __int64 v21; // rcx
  void *v22; // r12
  _QWORD *v23; // rax
  __int64 v24; // rdx
  _QWORD *v25; // r13
  __int64 v26; // rax
  __int64 *v27; // r8
  int v28; // eax
  __int64 v29; // rax
  unsigned __int16 *v30; // [rsp+0h] [rbp-F0h]
  unsigned __int16 *v32; // [rsp+10h] [rbp-E0h]
  __int64 v33; // [rsp+10h] [rbp-E0h]
  __int64 *v34; // [rsp+18h] [rbp-D8h]
  __int64 *v35; // [rsp+18h] [rbp-D8h]
  __int64 *v36; // [rsp+28h] [rbp-C8h] BYREF
  _DWORD *v37; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v38; // [rsp+38h] [rbp-B8h]
  _DWORD v39[44]; // [rsp+40h] [rbp-B0h] BYREF

  v37 = v39;
  v39[0] = a3;
  v38 = 0x2000000001LL;
  if ( (_DWORD)a3 )
  {
    v4 = (unsigned int)(a3 - 1);
    v5 = 32;
    v6 = (__int64)&a2[8 * v4 + 8];
    v7 = a2;
    v8 = 1;
    while ( 1 )
    {
      v9 = *v7;
      if ( !(_WORD)v9 )
        v9 = *((_QWORD *)v7 + 1);
      v10 = (unsigned int)v9;
      if ( v8 + 1 > v5 )
      {
        v30 = v7;
        sub_C8D5F0((__int64)&v37, v39, v8 + 1, 4u, (__int64)v7, (unsigned int)v9);
        v8 = (unsigned int)v38;
        v7 = v30;
        v10 = (unsigned int)v9;
      }
      v11 = HIDWORD(v9);
      v37[v8] = v10;
      LODWORD(v38) = v38 + 1;
      v12 = (unsigned int)v38;
      if ( (unsigned __int64)(unsigned int)v38 + 1 > HIDWORD(v38) )
      {
        v32 = v7;
        sub_C8D5F0((__int64)&v37, v39, (unsigned int)v38 + 1LL, 4u, (__int64)v7, v10);
        v12 = (unsigned int)v38;
        v7 = v32;
      }
      v7 += 8;
      v37[v12] = v11;
      v8 = (unsigned int)(v38 + 1);
      LODWORD(v38) = v38 + 1;
      if ( (unsigned __int16 *)v6 == v7 )
        break;
      v5 = HIDWORD(v38);
    }
  }
  v36 = 0;
  v13 = sub_C65B40((__int64)(a1 + 22), (__int64)&v37, (__int64 *)&v36, (__int64)off_4A367B0);
  v14 = a1 + 22;
  v15 = v13;
  if ( !v13 )
  {
    v18 = 16LL * (unsigned int)a3;
    v19 = a1[24];
    a1[34] += v18;
    v20 = a1 + 24;
    v21 = v18 + ((v19 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( a1[25] >= v21 && v19 )
    {
      a1[24] = v21;
      v22 = (void *)((v19 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    }
    else
    {
      v29 = sub_9D1E70((__int64)(a1 + 24), v18, 16LL * (unsigned int)a3, 3);
      v14 = a1 + 22;
      v22 = (void *)v29;
    }
    if ( 16 * a3 )
    {
      v34 = v14;
      memmove(v22, a2, 16 * a3);
      v14 = v34;
    }
    v35 = v14;
    v23 = sub_C65D30((__int64)&v37, v20);
    v33 = v24;
    v25 = v23;
    v26 = sub_A777F0(0x28u, (__int64 *)v20);
    v27 = v35;
    v15 = (_QWORD *)v26;
    if ( v26 )
    {
      *(_QWORD *)v26 = 0;
      *(_QWORD *)(v26 + 8) = v25;
      *(_QWORD *)(v26 + 16) = v33;
      *(_QWORD *)(v26 + 24) = v22;
      *(_DWORD *)(v26 + 32) = a3;
      v28 = sub_939680(v25, (__int64)v25 + 4 * v33);
      v27 = v35;
      *((_DWORD *)v15 + 9) = v28;
    }
    sub_C657C0(v27, v15, v36, (__int64)off_4A367B0);
  }
  v16 = v15[3];
  if ( v37 != v39 )
    _libc_free((unsigned __int64)v37);
  return v16;
}
