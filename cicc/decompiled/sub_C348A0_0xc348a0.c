// Function: sub_C348A0
// Address: 0xc348a0
//
__int64 __fastcall sub_C348A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char v3; // cl
  __int64 v4; // r15
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 *v8; // rax
  __int64 *v9; // rax
  __int64 v10; // r14
  _QWORD v11[8]; // [rsp+0h] [rbp-40h] BYREF

  v2 = *(unsigned __int8 *)(a2 + 20);
  v3 = *(_BYTE *)(a2 + 20) & 7;
  if ( v3 == 1 )
  {
    v6 = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 32766;
    v8 = (__int64 *)sub_C33930(a2);
    v4 = *v8;
    v5 = v8[1] & 0xFFFFFFFFFFFFLL;
    v2 = *(unsigned __int8 *)(a2 + 20);
  }
  else if ( v3 )
  {
    if ( v3 == 3 )
    {
      v4 = 0;
      v5 = 0;
      v6 = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) - 1;
    }
    else
    {
      LODWORD(v6) = *(_DWORD *)(a2 + 16) + (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 16382;
      v9 = (__int64 *)sub_C33930(a2);
      v6 = (int)v6;
      v4 = *v9;
      v10 = v9[1];
      if ( (int)v6 == 1 )
        v6 = *(_WORD *)(sub_C33930(a2) + 14) & 1;
      v5 = v10 & 0xFFFFFFFFFFFFLL;
      v2 = *(unsigned __int8 *)(a2 + 20);
    }
  }
  else
  {
    v6 = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 32766;
    v4 = 0;
    v5 = 0;
  }
  LOBYTE(v2) = (unsigned __int8)v2 >> 3;
  v11[0] = v4;
  v11[1] = (v6 << 48) & 0x7FFF000000000000LL | v5 | (v2 << 63);
  sub_C438C0(a1, 128, v11, 2);
  return a1;
}
