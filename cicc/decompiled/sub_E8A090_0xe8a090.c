// Function: sub_E8A090
// Address: 0xe8a090
//
unsigned __int64 __fastcall sub_E8A090(__int64 a1, size_t a2, size_t a3, __int64 a4)
{
  __int64 v4; // rbp
  _DWORD *v5; // r10
  unsigned __int64 v6; // rax
  _QWORD *v7; // rdi
  unsigned int v8; // r11d
  __int64 v9; // r8
  int v10; // ecx
  unsigned __int64 *v12; // rdi
  size_t v13[4]; // [rsp-68h] [rbp-68h] BYREF
  __int16 v14; // [rsp-48h] [rbp-48h]
  _QWORD v15[4]; // [rsp-38h] [rbp-38h] BYREF
  __int16 v16; // [rsp-18h] [rbp-18h]
  __int64 v17; // [rsp-8h] [rbp-8h]

  v5 = *(_DWORD **)(a1 + 920);
  if ( *v5 == 1 )
  {
    v17 = v4;
    if ( a4 )
    {
      v6 = *(_QWORD *)(a4 + 168) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v6 )
      {
LABEL_4:
        v7 = 0;
        v8 = 131;
LABEL_5:
        v9 = *(_QWORD *)(a4 + 16);
        v10 = *(_DWORD *)(a4 + 156);
        v15[0] = v7;
        v15[1] = v6;
        v16 = 261;
        v13[0] = a2;
        v13[1] = a3;
        v14 = 261;
        return sub_E71CB0((__int64)v5, v13, 1, v8, 0, (__int64)v15, 1u, v10, v9);
      }
    }
    else
    {
      a4 = *(_QWORD *)(a1 + 24);
      v6 = *(_QWORD *)(a4 + 168) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v6 )
        goto LABEL_4;
    }
    if ( (*(_BYTE *)(v6 + 8) & 1) != 0 )
    {
      v12 = *(unsigned __int64 **)(v6 - 8);
      v6 = *v12;
      v7 = v12 + 3;
    }
    else
    {
      v6 = 0;
      v7 = 0;
    }
    v8 = 643;
    goto LABEL_5;
  }
  return 0;
}
