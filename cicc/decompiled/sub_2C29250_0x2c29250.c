// Function: sub_2C29250
// Address: 0x2c29250
//
__int64 __fastcall sub_2C29250(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int16 v4; // bx
  __int64 v5; // r9
  __int16 v6; // r8
  __int64 v7; // rdi
  __int16 v8; // cx
  __int16 v9; // dx
  __int64 v10; // r11
  __int16 v11; // r10
  __int16 v12; // si

  result = a1;
  v3 = a2[6];
  v4 = *((_WORD *)a2 + 28);
  v5 = *a2;
  v6 = *((_WORD *)a2 + 4);
  v7 = a2[2];
  v8 = *((_WORD *)a2 + 12);
  v9 = *((_WORD *)a2 + 20);
  v10 = a2[8];
  v11 = *((_WORD *)a2 + 36);
  v12 = *((_WORD *)a2 + 44);
  *(_QWORD *)result = v3;
  *(_WORD *)(result + 8) = v4;
  *(_QWORD *)(result + 16) = v10;
  *(_WORD *)(result + 24) = v11;
  *(_WORD *)(result + 40) = v12;
  *(_QWORD *)(result + 48) = v5;
  *(_WORD *)(result + 56) = v6;
  *(_QWORD *)(result + 64) = v7;
  *(_WORD *)(result + 72) = v8;
  *(_WORD *)(result + 88) = v9;
  return result;
}
