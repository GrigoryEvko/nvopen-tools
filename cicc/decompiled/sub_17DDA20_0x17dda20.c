// Function: sub_17DDA20
// Address: 0x17dda20
//
unsigned __int64 __fastcall sub_17DDA20(_QWORD *a1, __int64 a2)
{
  char v3; // cl
  _QWORD *v4; // rbx
  __int64 v5; // rax
  __int128 v6; // rdi
  _QWORD *v7; // rdx
  __int64 v8; // rax
  _QWORD *v9; // r14
  __int64 *v10; // rax
  unsigned __int64 result; // rax
  _BYTE *v12; // [rsp+0h] [rbp-90h] BYREF
  __int64 v13; // [rsp+8h] [rbp-88h]
  __int64 *v14; // [rsp+10h] [rbp-80h]
  _QWORD *v15; // [rsp+18h] [rbp-78h]
  __int64 v16[14]; // [rsp+20h] [rbp-70h] BYREF

  sub_17CE510((__int64)v16, a2, 0, 0, 0);
  v3 = *(_BYTE *)(a2 + 23);
  v14 = v16;
  v12 = 0;
  v13 = 0;
  v15 = a1;
  if ( (v3 & 0x40) != 0 )
  {
    v4 = *(_QWORD **)(a2 - 8);
    v5 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  }
  else
  {
    v5 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v4 = (_QWORD *)(a2 - 24 * v5);
  }
  while ( 1 )
  {
    v7 = (_QWORD *)a2;
    v8 = 24 * v5;
    if ( (v3 & 0x40) != 0 )
      v7 = (_QWORD *)(*(_QWORD *)(a2 - 8) + v8);
    if ( v4 == v7 )
      break;
    *((_QWORD *)&v6 + 1) = *v4;
    *(_QWORD *)&v6 = &v12;
    v4 += 3;
    sub_17D7560(v6);
    v3 = *(_BYTE *)(a2 + 23);
    v5 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  }
  v9 = v15;
  v10 = sub_17CD8D0(v15, *(_QWORD *)a2);
  v12 = (_BYTE *)sub_17CF940(v9, v14, v12, (__int64)v10, 0);
  sub_17D4920((__int64)v15, (__int64 *)a2, (__int64)v12);
  result = *(unsigned int *)(v15[1] + 156LL);
  if ( (_DWORD)result )
    result = sub_17D4B80((__int64)v15, a2, v13);
  if ( v16[0] )
    return sub_161E7C0((__int64)v16, v16[0]);
  return result;
}
