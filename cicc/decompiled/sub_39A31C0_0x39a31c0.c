// Function: sub_39A31C0
// Address: 0x39a31c0
//
__int64 __fastcall sub_39A31C0(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int16 v3; // r14
  __int16 v4; // r13
  __int64 v5; // r12
  __int64 v6; // r12
  __int64 result; // rax
  unsigned __int64 v8; // rsi
  unsigned __int64 *v9; // rdx
  __int64 v10; // r12
  __int64 v11; // r12
  __int64 v12; // r12
  __int64 v13; // r12
  __int64 v14; // r12
  __int64 v15; // r12
  __int64 v16; // r12
  __int64 v17; // r12
  __int64 v18; // r12

  v3 = *((_WORD *)a3 + 2);
  v4 = *((_WORD *)a3 + 3);
  v5 = *a3;
  switch ( *(_DWORD *)a3 )
  {
    case 1:
      v10 = a3[1];
      result = sub_145CBF0(a2, 24, 16);
      *(_WORD *)(result + 12) = v3;
      *(_DWORD *)(result + 8) = 1;
      v8 = result & 0xFFFFFFFFFFFFFFFBLL;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v4;
      *(_QWORD *)(result + 16) = v10;
      break;
    case 2:
      v11 = a3[1];
      result = sub_145CBF0(a2, 24, 16);
      *(_WORD *)(result + 12) = v3;
      *(_DWORD *)(result + 8) = 2;
      v8 = result & 0xFFFFFFFFFFFFFFFBLL;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v4;
      *(_QWORD *)(result + 16) = v11;
      break;
    case 3:
      v12 = a3[1];
      result = sub_145CBF0(a2, 24, 16);
      *(_WORD *)(result + 12) = v3;
      *(_DWORD *)(result + 8) = 3;
      v8 = result & 0xFFFFFFFFFFFFFFFBLL;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v4;
      *(_QWORD *)(result + 16) = v12;
      break;
    case 4:
      v13 = a3[1];
      result = sub_145CBF0(a2, 24, 16);
      *(_WORD *)(result + 12) = v3;
      *(_DWORD *)(result + 8) = 4;
      v8 = result & 0xFFFFFFFFFFFFFFFBLL;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v4;
      *(_QWORD *)(result + 16) = v13;
      break;
    case 5:
      v14 = a3[1];
      result = sub_145CBF0(a2, 24, 16);
      *(_WORD *)(result + 12) = v3;
      *(_DWORD *)(result + 8) = 5;
      v8 = result & 0xFFFFFFFFFFFFFFFBLL;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v4;
      *(_QWORD *)(result + 16) = v14;
      break;
    case 6:
      v15 = a3[1];
      result = sub_145CBF0(a2, 24, 16);
      *(_WORD *)(result + 12) = v3;
      *(_DWORD *)(result + 8) = 6;
      v8 = result & 0xFFFFFFFFFFFFFFFBLL;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v4;
      *(_QWORD *)(result + 16) = v15;
      break;
    case 7:
      v16 = a3[1];
      result = sub_145CBF0(a2, 24, 16);
      *(_WORD *)(result + 12) = v3;
      *(_DWORD *)(result + 8) = 7;
      v8 = result & 0xFFFFFFFFFFFFFFFBLL;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v4;
      *(_QWORD *)(result + 16) = v16;
      break;
    case 8:
      v17 = a3[1];
      result = sub_145CBF0(a2, 24, 16);
      *(_WORD *)(result + 12) = v3;
      *(_DWORD *)(result + 8) = 8;
      v8 = result & 0xFFFFFFFFFFFFFFFBLL;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v4;
      *(_QWORD *)(result + 16) = v17;
      break;
    case 9:
      v18 = a3[1];
      result = sub_145CBF0(a2, 24, 16);
      *(_WORD *)(result + 12) = v3;
      *(_DWORD *)(result + 8) = 9;
      v8 = result & 0xFFFFFFFFFFFFFFFBLL;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v4;
      *(_QWORD *)(result + 16) = v18;
      break;
    case 0xA:
      v6 = a3[1];
      result = sub_145CBF0(a2, 24, 16);
      *(_DWORD *)(result + 8) = 10;
      *(_WORD *)(result + 12) = v3;
      v8 = result & 0xFFFFFFFFFFFFFFFBLL;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v4;
      *(_QWORD *)(result + 16) = v6;
      break;
    default:
      result = sub_145CBF0(a2, 24, 16);
      *(_QWORD *)(result + 8) = v5;
      v8 = result & 0xFFFFFFFFFFFFFFFBLL;
      *(_QWORD *)result = result | 4;
      break;
  }
  v9 = (unsigned __int64 *)*a1;
  if ( *a1 )
  {
    *(_QWORD *)result = *v9;
    *v9 = v8;
  }
  *a1 = result;
  return result;
}
