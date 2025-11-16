// Function: sub_38D59D0
// Address: 0x38d59d0
//
void __fastcall sub_38D59D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int64 v6; // r14
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 *v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  char v19; // dl
  __int64 v20; // rax
  unsigned __int64 v21; // [rsp+8h] [rbp-38h] BYREF

  if ( a3 )
  {
    v6 = sub_38D3D10(a1, a4, a3);
    v7 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 72LL);
    if ( v7 == sub_38D3BD0 )
    {
      LODWORD(v8) = 0;
      if ( *(_BYTE *)(a1 + 260) )
        v8 = *(_QWORD *)(a1 + 264);
    }
    else
    {
      LODWORD(v8) = v7(a1);
    }
    if ( sub_38CF2B0(v6, &v21, v8) )
    {
      sub_38C6B90(
        (_QWORD *)a1,
        *(unsigned __int16 *)(*(_QWORD *)(a1 + 264) + 176LL)
      | (*(unsigned __int8 *)(*(_QWORD *)(a1 + 264) + 178LL) << 16),
        a2,
        v21);
    }
    else
    {
      v9 = sub_22077B0(0x90u);
      v10 = v9;
      if ( v9 )
      {
        v11 = v9;
        sub_38CF760(v9, 6, 0, 0);
        *(_QWORD *)(v10 + 56) = 0;
        *(_WORD *)(v10 + 48) = 0;
        *(_QWORD *)(v10 + 64) = v10 + 80;
        *(_QWORD *)(v10 + 72) = 0x800000000LL;
        *(_QWORD *)(v10 + 88) = v10 + 104;
        *(_QWORD *)(v10 + 96) = 0x100000000LL;
        *(_QWORD *)(v10 + 128) = a2;
        *(_QWORD *)(v10 + 136) = v6;
      }
      else
      {
        v11 = 0;
      }
      sub_38D4150(a1, v10, 0);
      v12 = *(unsigned int *)(a1 + 120);
      v13 = 0;
      if ( (_DWORD)v12 )
        v13 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v12 - 32);
      v14 = *(__int64 **)(a1 + 272);
      v15 = *v14;
      v16 = *(_QWORD *)v10 & 7LL;
      *(_QWORD *)(v10 + 8) = v14;
      v15 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v10 = v15 | v16;
      *(_QWORD *)(v15 + 8) = v11;
      *v14 = *v14 & 7 | v11;
      *(_QWORD *)(v10 + 24) = v13;
    }
  }
  else
  {
    v17 = *(_QWORD *)(a1 + 264);
    v19 = *(_BYTE *)(v17 + 178);
    LOWORD(v21) = *(_WORD *)(v17 + 176);
    v20 = *(_QWORD *)a1;
    BYTE2(v21) = v19;
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(v20 + 424))(a1, 0, 1);
    sub_38DCDD0(a1, (int)(a5 + 1));
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 424LL))(a1, 2, 1);
    sub_38DDC80(a1, a4, a5);
    sub_38C6B90((_QWORD *)a1, v21, a2, 0);
  }
}
