// Function: sub_23CF050
// Address: 0x23cf050
//
__int64 __fastcall sub_23CF050(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 result; // rax
  __int64 v4[5]; // [rsp+8h] [rbp-28h] BYREF

  v4[0] = sub_B2D7E0(a2, "unsafe-fp-math", 0xEu);
  *(_BYTE *)(a1 + 864) = sub_A72A30(v4) & 1 | *(_BYTE *)(a1 + 864) & 0xFE;
  v4[0] = sub_B2D7E0(a2, "no-infs-fp-math", 0xFu);
  *(_BYTE *)(a1 + 864) = (2 * (sub_A72A30(v4) & 1)) | *(_BYTE *)(a1 + 864) & 0xFD;
  v4[0] = sub_B2D7E0(a2, "no-nans-fp-math", 0xFu);
  *(_BYTE *)(a1 + 864) = (4 * (sub_A72A30(v4) & 1)) | *(_BYTE *)(a1 + 864) & 0xFB;
  v4[0] = sub_B2D7E0(a2, "no-signed-zeros-fp-math", 0x17u);
  *(_BYTE *)(a1 + 864) = (16 * (sub_A72A30(v4) & 1)) | *(_BYTE *)(a1 + 864) & 0xEF;
  v4[0] = sub_B2D7E0(a2, "approx-func-fp-math", 0x13u);
  v2 = 32 * (sub_A72A30(v4) & 1);
  result = v2 | *(_BYTE *)(a1 + 864) & 0xDFu;
  *(_BYTE *)(a1 + 864) = v2 | *(_BYTE *)(a1 + 864) & 0xDF;
  return result;
}
