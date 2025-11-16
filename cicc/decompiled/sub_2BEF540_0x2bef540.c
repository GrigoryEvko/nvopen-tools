// Function: sub_2BEF540
// Address: 0x2bef540
//
__int64 __fastcall sub_2BEF540(__int64 a1, __int64 *a2)
{
  __int64 v2; // r8
  __int64 v3; // rcx
  __int64 result; // rax
  __int16 v5; // dx
  __int16 v6; // di
  __int64 v7; // r11
  __int16 v8; // r10
  __int64 v9; // r9
  __int16 v10; // si

  v2 = a2[5];
  v3 = a2[7];
  result = a1;
  v5 = *((_WORD *)a2 + 32);
  v6 = *((_WORD *)a2 + 24);
  v7 = *a2;
  v8 = *((_WORD *)a2 + 4);
  v9 = a2[2];
  v10 = *((_WORD *)a2 + 12);
  *(_BYTE *)(result + 41) = 1;
  *(_QWORD *)result = v7;
  *(_WORD *)(result + 8) = v8;
  *(_QWORD *)(result + 16) = v9;
  *(_WORD *)(result + 24) = v10;
  *(_QWORD *)(result + 48) = v2;
  *(_WORD *)(result + 56) = v6;
  *(_QWORD *)(result + 64) = v3;
  *(_WORD *)(result + 72) = v5;
  *(_BYTE *)(result + 89) = 1;
  return result;
}
