// Function: sub_28ECBA0
// Address: 0x28ecba0
//
__int64 __fastcall sub_28ECBA0(__int64 a1, unsigned int a2, unsigned int a3, __int64 *a4)
{
  unsigned int v5; // eax
  __int64 v6; // rdx
  _QWORD *v11; // rsi
  __int64 v12; // rax
  _QWORD *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r10
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+8h] [rbp-68h]
  unsigned __int64 v20[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v21; // [rsp+20h] [rbp-50h]
  char v22; // [rsp+30h] [rbp-40h]
  char v23; // [rsp+31h] [rbp-3Fh]

  v5 = *((_DWORD *)a4 + 2);
  v6 = *a4;
  if ( v5 == 1 )
    return *(_QWORD *)(v6 + 16);
  v20[0] = 6;
  v20[1] = 0;
  v11 = (_QWORD *)(v6 + 24LL * v5 - 24);
  v21 = v11[2];
  if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
  {
    sub_BD6050(v20, *v11 & 0xFFFFFFFFFFFFFFF8LL);
    v5 = *((_DWORD *)a4 + 2);
    v6 = *a4;
  }
  v12 = v5 - 1;
  *((_DWORD *)a4 + 2) = v12;
  v13 = (_QWORD *)(v6 + 24 * v12);
  v14 = v13[2];
  if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
    sub_BD60C0(v13);
  v15 = v21;
  if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
  {
    v18 = v21;
    sub_BD60C0(v20);
    v15 = v18;
  }
  v19 = v15;
  v16 = sub_28ECBA0(a1, a2, a3, a4);
  v17 = a1 - 24;
  if ( !a1 )
    v17 = 0;
  v23 = 1;
  v20[0] = (unsigned __int64)"reass.add";
  v22 = 3;
  return sub_28E9200(v16, v19, (__int64)v20, a1, a2, a3, v17);
}
