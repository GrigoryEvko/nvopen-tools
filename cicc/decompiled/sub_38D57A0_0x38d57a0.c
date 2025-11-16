// Function: sub_38D57A0
// Address: 0x38d57a0
//
unsigned __int64 __fastcall sub_38D57A0(__int64 a1)
{
  unsigned __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax

  result = sub_38D4B30(a1);
  if ( !result || *(_BYTE *)(result + 16) != 9 )
  {
    v2 = sub_22077B0(0x108u);
    v3 = v2;
    if ( v2 )
    {
      v4 = v2;
      sub_38CF760(v2, 9, 0, 0);
      *(_QWORD *)(v3 + 48) = 0;
      *(_BYTE *)(v3 + 56) = 0;
      *(_QWORD *)(v3 + 64) = 0;
      *(_BYTE *)(v3 + 72) = 0;
      memset((void *)(v3 + 80), 0, 0xA8u);
      *(_BYTE *)(v3 + 248) = 0;
      *(_QWORD *)(v3 + 96) = v3 + 112;
      *(_QWORD *)(v3 + 104) = 0x800000000LL;
      *(_QWORD *)(v3 + 256) = 0;
    }
    else
    {
      v4 = 0;
    }
    sub_38D4150(a1, v3, 0);
    v5 = *(unsigned int *)(a1 + 120);
    v6 = 0;
    if ( (_DWORD)v5 )
      v6 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v5 - 32);
    v7 = *(__int64 **)(a1 + 272);
    v8 = *v7;
    v9 = *(_QWORD *)v3 & 7LL;
    *(_QWORD *)(v3 + 8) = v7;
    v8 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v3 = v8 | v9;
    *(_QWORD *)(v8 + 8) = v4;
    *v7 = *v7 & 7 | v4;
    *(_QWORD *)(v3 + 24) = v6;
    return v3;
  }
  return result;
}
