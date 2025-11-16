// Function: sub_E6CBD0
// Address: 0xe6cbd0
//
unsigned __int64 __fastcall sub_E6CBD0(
        _QWORD *a1,
        const void *a2,
        size_t a3,
        int a4,
        unsigned int a5,
        int a6,
        __int64 a7,
        unsigned __int8 a8,
        int a9,
        __int64 a10)
{
  _QWORD *v13; // rbx
  __int64 v14; // rax
  int v15; // r10d
  int v16; // r11d
  unsigned __int64 v17; // r12
  unsigned __int64 v18; // rdi
  __int64 v20; // rax
  int v21; // [rsp+10h] [rbp-40h]

  v21 = a3;
  v13 = sub_E6CA50((__int64)a1, a2, a3);
  sub_EA1710(v13, 0);
  sub_EA15B0(v13, 3);
  v14 = a1[72];
  v15 = v21;
  a1[82] += 200LL;
  v16 = (int)a2;
  v17 = (v14 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[73] >= v17 + 200 && v14 )
  {
    a1[72] = v17 + 200;
  }
  else
  {
    v20 = sub_9D1E70((__int64)(a1 + 72), 200, 200, 3);
    v15 = v21;
    v16 = (int)a2;
    v17 = v20;
  }
  sub_E92760(v17, 1, v16, v15, (a5 >> 2) & 1, a4 == 8, (__int64)v13);
  *(_DWORD *)(v17 + 148) = a4;
  *(_DWORD *)(v17 + 152) = a5;
  *(_QWORD *)v17 = &unk_49E3600;
  *(_DWORD *)(v17 + 156) = a9;
  *(_DWORD *)(v17 + 160) = a6;
  *(_QWORD *)(v17 + 168) = (4LL * a8) | a7 & 0xFFFFFFFFFFFFFFFBLL;
  *(_QWORD *)(v17 + 176) = a10;
  v18 = (unsigned __int16)(4 * a8) & 0xFFF8 | a7 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v18 )
    ((void (__fastcall *)(unsigned __int64, __int64))sub_EA16E0)(v18, 1);
  *v13 = sub_E6B260(a1, v17);
  return v17;
}
