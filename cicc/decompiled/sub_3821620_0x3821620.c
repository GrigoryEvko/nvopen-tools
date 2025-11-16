// Function: sub_3821620
// Address: 0x3821620
//
void __fastcall sub_3821620(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  int v8; // eax
  unsigned __int64 *v9; // rax
  __int64 v10; // rdx
  unsigned __int16 *v11; // rax
  __int64 v12; // r9
  int v13; // edx
  __int64 v14; // rdx
  unsigned __int16 *v15; // rax
  __int64 v16; // r9
  unsigned __int8 *v17; // rax
  __int64 v18; // rsi
  int v19; // edx
  int v20; // [rsp+18h] [rbp-88h]
  __int64 v21; // [rsp+20h] [rbp-80h] BYREF
  int v22; // [rsp+28h] [rbp-78h]
  __int128 v23; // [rsp+30h] [rbp-70h] BYREF
  __int128 v24; // [rsp+40h] [rbp-60h] BYREF
  __int128 v25; // [rsp+50h] [rbp-50h] BYREF
  __int128 v26; // [rsp+60h] [rbp-40h] BYREF

  v7 = *(_QWORD *)(a2 + 80);
  v21 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v21, v7, 1);
  v8 = *(_DWORD *)(a2 + 72);
  DWORD2(v23) = 0;
  DWORD2(v24) = 0;
  v22 = v8;
  v9 = *(unsigned __int64 **)(a2 + 40);
  DWORD2(v25) = 0;
  DWORD2(v26) = 0;
  v10 = v9[1];
  *(_QWORD *)&v23 = 0;
  *(_QWORD *)&v24 = 0;
  *(_QWORD *)&v25 = 0;
  *(_QWORD *)&v26 = 0;
  sub_375E510(a1, *v9, v10, (__int64)&v23, (__int64)&v24);
  sub_375E510(
    a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
    (__int64)&v25,
    (__int64)&v26);
  v11 = (unsigned __int16 *)(*(_QWORD *)(v23 + 48) + 16LL * DWORD2(v23));
  *(_QWORD *)a3 = sub_3406EB0(
                    *(_QWORD **)(a1 + 8),
                    *(_DWORD *)(a2 + 24),
                    (__int64)&v21,
                    *v11,
                    *((_QWORD *)v11 + 1),
                    v12,
                    v23,
                    v25);
  v20 = v13;
  v14 = v23;
  *(_DWORD *)(a3 + 8) = v20;
  v15 = (unsigned __int16 *)(*(_QWORD *)(v14 + 48) + 16LL * DWORD2(v23));
  v17 = sub_3406EB0(
          *(_QWORD **)(a1 + 8),
          *(_DWORD *)(a2 + 24),
          (__int64)&v21,
          *v15,
          *((_QWORD *)v15 + 1),
          v16,
          v24,
          v26);
  v18 = v21;
  *(_QWORD *)a4 = v17;
  *(_DWORD *)(a4 + 8) = v19;
  if ( v18 )
    sub_B91220((__int64)&v21, v18);
}
