// Function: sub_2E7AAE0
// Address: 0x2e7aae0
//
__int64 __fastcall sub_2E7AAE0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r12
  __int64 v5; // rax
  int v7; // eax
  int v8; // edx
  bool v9; // zf
  __int64 v10; // rcx
  unsigned __int64 v11; // rax

  v4 = *(_QWORD *)(a1 + 312);
  if ( v4 )
  {
    *(_QWORD *)(a1 + 312) = *(_QWORD *)v4;
LABEL_3:
    sub_2E30860(v4, a1, a2);
    goto LABEL_4;
  }
  v10 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 208) += 288LL;
  v11 = (v10 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(a1 + 136) < v11 + 288 || !v10 )
  {
    v11 = sub_9D1E70(a1 + 128, 288, 288, 3);
LABEL_17:
    v4 = v11;
    goto LABEL_3;
  }
  *(_QWORD *)(a1 + 128) = v11 + 288;
  if ( v11 )
    goto LABEL_17;
LABEL_4:
  v5 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(v5 + 879) & 0x10) == 0 && *(_DWORD *)(v5 + 880) != 1 )
    return v4;
  if ( a4 )
  {
    v7 = a3;
    v8 = HIDWORD(a3);
  }
  else
  {
    v7 = *(_DWORD *)(a1 + 584);
    *(_DWORD *)(a1 + 584) = v7 + 1;
    v8 = 0;
  }
  v9 = *(_BYTE *)(v4 + 248) == 0;
  *(_DWORD *)(v4 + 240) = v7;
  *(_DWORD *)(v4 + 244) = v8;
  if ( !v9 )
    return v4;
  *(_BYTE *)(v4 + 248) = 1;
  return v4;
}
