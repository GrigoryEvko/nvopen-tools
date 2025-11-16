// Function: sub_8A4520
// Address: 0x8a4520
//
__int64 __fastcall sub_8A4520(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int16 a5, _DWORD *a6, __int64 a7)
{
  __int64 v8; // r12
  __int64 v12; // rax
  char v13; // dl
  __int64 *v14; // rax
  __int64 *v15; // r13
  __int64 v16; // r14
  _QWORD *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 *v20; // rdi
  __int64 v21; // [rsp+8h] [rbp-38h] BYREF
  int v22[9]; // [rsp+1Ch] [rbp-24h] BYREF

  v8 = a1;
  v21 = a2;
  if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
  {
    v12 = (__int64)sub_8A1CE0(*(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), v21, a3, a4, 0, 0, a5, a6, a7);
    if ( v12 )
    {
      v13 = *(_BYTE *)(v12 + 80);
      if ( v13 == 16 )
      {
        v12 = **(_QWORD **)(v12 + 88);
        v13 = *(_BYTE *)(v12 + 80);
      }
      if ( v13 == 24 )
      {
        v12 = *(_QWORD *)(v12 + 88);
        if ( v12 && *(_BYTE *)(v12 + 80) == 19 )
          goto LABEL_10;
      }
      else if ( v13 == 19 )
      {
LABEL_10:
        v8 = *(_QWORD *)(*(_QWORD *)(v12 + 88) + 104LL);
        goto LABEL_2;
      }
    }
    *a6 = 1;
    v12 = sub_87F550();
    goto LABEL_10;
  }
LABEL_2:
  if ( (*(_BYTE *)(sub_8794A0((_QWORD *)v8) + 266) & 1) != 0 )
  {
    v14 = sub_8A4460((unsigned int *)(v8 + 128), a5, &v21, a3);
    v15 = v14;
    if ( v14 && (v16 = v14[4]) != 0 )
    {
      v17 = *(_QWORD **)(a7 + 64);
      if ( v17 )
      {
        v18 = v17[2];
        v19 = *(unsigned int *)(v8 + 128);
        v20 = *(__int64 **)(a7 + 64);
        v22[0] = 0;
        if ( v19 < v18 )
          v19 = v18;
        sub_89F5F0(v20, v19, v22);
        *(_DWORD *)(*v17 + 4LL * (unsigned int)(*(_DWORD *)(v8 + 128) - 1)) = 1;
      }
      v8 = v16;
      if ( (v15[3] & 0x10) != 0 )
        *(_DWORD *)(a7 + 88) = 1;
    }
    else if ( (a5 & 0x2000) != 0 )
    {
      *(_DWORD *)(a7 + 88) |= (*(_BYTE *)(v8 + 121) & 4) != 0;
    }
  }
  return v8;
}
