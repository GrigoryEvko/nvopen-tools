// Function: sub_39F1100
// Address: 0x39f1100
//
unsigned __int64 __fastcall sub_39F1100(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v7; // rdi
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 (__fastcall *v10)(__int64, unsigned int, __int64, int, unsigned int); // rax
  char v11; // al
  char v12; // cl
  unsigned __int64 result; // rax
  __int64 *v14; // rax
  int v15; // eax
  __int64 *v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // [rsp+0h] [rbp-90h]
  __int64 v20; // [rsp+8h] [rbp-88h]
  _QWORD v21[2]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v22[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v23; // [rsp+30h] [rbp-60h]
  _QWORD v24[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v25; // [rsp+50h] [rbp-40h]

  sub_390D5F0(a1[33], a2, 0);
  if ( !sub_38E27B0(a2) )
  {
    sub_38E2920(a2, 1u);
    *(_BYTE *)(a2 + 8) |= 0x10u;
  }
  sub_38E28A0(a2, 1);
  if ( (unsigned int)sub_38E27C0(a2) )
  {
    v11 = *(_BYTE *)(a2 + 9);
    if ( (v11 & 0xC) == 0xC )
    {
      if ( a3 != *(_QWORD *)(a2 + 24) )
        goto LABEL_22;
      v15 = *(_DWORD *)(a2 + 8) & 0x1F000;
      if ( v15 )
        v15 = 1 << (((*(_DWORD *)(a2 + 8) >> 12) & 0x1F) - 1);
      if ( a4 != v15 )
      {
LABEL_22:
        if ( (*(_BYTE *)a2 & 4) != 0 )
        {
          v16 = *(__int64 **)(a2 - 8);
          v17 = *v16;
          v18 = v16 + 2;
        }
        else
        {
          v17 = 0;
          v18 = 0;
        }
        v21[0] = v18;
        v22[0] = "Symbol: ";
        v22[1] = v21;
        v24[0] = v22;
        v21[1] = v17;
        v23 = 1283;
        v24[1] = " redeclared as different type";
        v25 = 770;
        sub_16BCFB0((__int64)v24, 1u);
      }
    }
    else
    {
      *(_QWORD *)(a2 + 24) = a3;
      v12 = 0;
      *(_BYTE *)(a2 + 9) = v11 | 0xC;
      if ( a4 )
      {
        _BitScanReverse(&a4, a4);
        v12 = -(a4 ^ 0x1F) & 0x1F;
      }
      *(_DWORD *)(a2 + 8) = *(_DWORD *)(a2 + 8) & 0xFFFE0FFF | ((v12 & 0x1F) << 12);
    }
  }
  else
  {
    v7 = *(_QWORD *)a1[33];
    v22[0] = ".bss";
    v23 = 259;
    v25 = 257;
    v8 = sub_38C3B80(v7, (__int64)v22, 8, 3, 0, (__int64)v24, -1, 0);
    v9 = *((unsigned int *)a1 + 30);
    if ( (_DWORD)v9 )
    {
      v14 = (__int64 *)(a1[14] + 32 * v9 - 32);
      v19 = *v14;
      v20 = v14[1];
    }
    else
    {
      v20 = 0;
      v19 = 0;
    }
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 160))(a1, v8, 0);
    v10 = *(__int64 (__fastcall **)(__int64, unsigned int, __int64, int, unsigned int))(*a1 + 512);
    if ( v10 == sub_39F10A0 )
    {
      if ( sub_39EF7F0((__int64)a1) )
        sub_16BD130("Emitting values inside a locked bundle is forbidden", 1u);
      sub_38D4400((__int64)a1, a4, 0, 1, 0);
    }
    else
    {
      v10((__int64)a1, a4, 0, 1, 0);
    }
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, a2, 0);
    sub_38DD110(a1, a3);
    if ( a4 > *(_DWORD *)(v8 + 24) )
      *(_DWORD *)(v8 + 24) = a4;
    (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 160))(a1, v19, v20);
  }
  result = sub_38CB470(a3, a1[1]);
  *(_QWORD *)(a2 + 32) = result;
  return result;
}
