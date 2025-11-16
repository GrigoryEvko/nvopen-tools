// Function: sub_22ABF70
// Address: 0x22abf70
//
__int64 __fastcall sub_22ABF70(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rbx
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rax
  bool v9; // zf
  __int64 v10; // rdx
  __int64 result; // rax

  v4 = (_QWORD *)sub_22077B0(0x80u);
  v5 = v4;
  if ( v4 )
  {
    v4[1] = 2;
    v4[2] = 0;
    v4[3] = a2;
    if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
      sub_BD73F0((__int64)(v4 + 1));
    v5[4] = 0;
    v5[5] = 0;
    v5[6] = a1;
    *v5 = &unk_4A09CC0;
    v5[7] = 6;
    v5[8] = 0;
    v5[9] = a3;
    if ( a3 == -4096 || a3 == 0 || a3 == -8192 )
    {
      v6 = 0;
    }
    else
    {
      sub_BD73F0((__int64)(v5 + 7));
      v6 = v5[4] & 7LL;
    }
    v5[10] = 0;
    v5[11] = v5 + 14;
    v5[12] = 2;
    *((_DWORD *)v5 + 26) = 0;
    *((_BYTE *)v5 + 108) = 1;
  }
  else
  {
    v6 = MEMORY[0x20] & 7;
  }
  v7 = *(_QWORD *)(a1 + 200);
  v5[5] = a1 + 200;
  v7 &= 0xFFFFFFFFFFFFFFF8LL;
  v5[4] = v7 | v6;
  *(_QWORD *)(v7 + 8) = v5 + 4;
  v8 = *(_QWORD *)(a1 + 200) & 7LL | (unsigned __int64)(v5 + 4);
  *(_QWORD *)(a1 + 200) = v8;
  v8 &= 0xFFFFFFFFFFFFFFF8LL;
  v9 = v8 == 0;
  v10 = v8 - 32;
  result = 0;
  if ( !v9 )
    return v10;
  return result;
}
