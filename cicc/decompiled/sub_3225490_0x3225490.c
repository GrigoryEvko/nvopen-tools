// Function: sub_3225490
// Address: 0x3225490
//
__int64 __fastcall sub_3225490(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r14
  __int64 result; // rax
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // r8
  void (*v9)(); // rax
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned __int8 v12; // dl
  unsigned __int16 v13; // ax
  __int64 v14; // r10
  __int64 v15; // rdi
  void (*v16)(); // rcx
  bool v17; // al
  __int64 v18; // r10
  bool v19; // zf
  __int64 v20; // rdi
  void (*v21)(); // rax
  __int64 v22; // r10
  __int64 v23; // rdi
  void (*v24)(); // rax
  __int64 v25; // [rsp+0h] [rbp-80h]
  __int64 v26; // [rsp+8h] [rbp-78h]
  __int64 v27; // [rsp+8h] [rbp-78h]
  __int64 v28; // [rsp+8h] [rbp-78h]
  unsigned __int16 v29; // [rsp+8h] [rbp-78h]
  __int64 v30; // [rsp+8h] [rbp-78h]
  __int64 *i; // [rsp+18h] [rbp-68h]
  _QWORD v32[4]; // [rsp+20h] [rbp-60h] BYREF
  char v33; // [rsp+40h] [rbp-40h]
  char v34; // [rsp+41h] [rbp-3Fh]

  v2 = *(__int64 **)(a1 + 656);
  v3 = 2LL * *(unsigned int *)(a1 + 664);
  result = (__int64)&v2[v3];
  for ( i = &v2[v3]; i != v2; v2 += 2 )
  {
    v10 = *(_QWORD *)(v2[1] + 408);
    if ( !v10 )
      v10 = v2[1];
    v11 = *v2;
    v12 = *(_BYTE *)(*v2 - 16);
    if ( (v12 & 2) != 0 )
      result = *(_QWORD *)(v11 - 32);
    else
      result = v11 - 16 - 8LL * ((v12 >> 2) & 0xF);
    v6 = *(_QWORD *)(result + 64);
    if ( v6 )
    {
      result = (*(_BYTE *)(v6 - 16) & 2) != 0 ? *(unsigned int *)(v6 - 24) : (*(_WORD *)(v6 - 16) >> 6) & 0xFu;
      if ( (_DWORD)result )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 176LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
          a2,
          0);
        (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 208LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
          *(_QWORD *)(v10 + 416),
          0);
        if ( *(_BYTE *)(a1 + 3692) )
        {
          v13 = sub_3220AA0(a1);
          v14 = *(_QWORD *)(a1 + 8);
          v15 = *(_QWORD *)(v14 + 224);
          v16 = *(void (**)())(*(_QWORD *)v15 + 120LL);
          v34 = 1;
          v32[0] = "Macro information version";
          v33 = 3;
          if ( v16 != nullsub_98 )
          {
            v25 = v14;
            v29 = v13;
            ((void (__fastcall *)(__int64, _QWORD *, __int64))v16)(v15, v32, 1);
            v14 = v25;
            v13 = v29;
          }
          v26 = v14;
          if ( v13 < 4u )
            v13 = 4;
          sub_31DC9F0(v14, v13);
          v17 = sub_31DF690(v26);
          v18 = v26;
          v19 = !v17;
          v20 = *(_QWORD *)(v26 + 224);
          v21 = *(void (**)())(*(_QWORD *)v20 + 120LL);
          v34 = 1;
          if ( v19 )
          {
            v33 = 3;
            v32[0] = "Flags: 32 bit, debug_line_offset present";
            if ( v21 != nullsub_98 )
            {
              ((void (__fastcall *)(__int64, _QWORD *, __int64))v21)(v20, v32, 1);
              v18 = v26;
            }
            v28 = v18;
            sub_31DC9D0(v18, 2);
            v22 = v28;
          }
          else
          {
            v33 = 3;
            v32[0] = "Flags: 64 bit, debug_line_offset present";
            if ( v21 != nullsub_98 )
            {
              ((void (__fastcall *)(__int64, _QWORD *, __int64))v21)(v20, v32, 1);
              v18 = v26;
            }
            v27 = v18;
            sub_31DC9D0(v18, 3);
            v22 = v27;
          }
          v23 = *(_QWORD *)(v22 + 224);
          v24 = *(void (**)())(*(_QWORD *)v23 + 120LL);
          v34 = 1;
          v32[0] = "debug_line_offset";
          v33 = 3;
          if ( v24 != nullsub_98 )
          {
            v30 = v22;
            ((void (__fastcall *)(__int64, _QWORD *, __int64))v24)(v23, v32, 1);
            v22 = v30;
          }
          if ( *(_BYTE *)(a1 + 3769) )
            sub_31F0F00(v22, 0);
          else
            sub_31F0D70(v22, *(_QWORD *)(v10 + 400), 0);
        }
        sub_32253E0(a1, v6, v10);
        v7 = *(_QWORD *)(a1 + 8);
        v8 = *(_QWORD *)(v7 + 224);
        v9 = *(void (**)())(*(_QWORD *)v8 + 120LL);
        v34 = 1;
        v32[0] = "End Of Macro List Mark";
        v33 = 3;
        if ( v9 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v9)(v8, v32, 1);
          v7 = *(_QWORD *)(a1 + 8);
        }
        result = sub_31DC9D0(v7, 0);
      }
    }
  }
  return result;
}
