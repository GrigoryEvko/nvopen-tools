// Function: sub_3259990
// Address: 0x3259990
//
void __fastcall sub_3259990(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  _BYTE *v6; // rcx
  _QWORD *v7; // rdx
  unsigned __int8 *v8; // rax
  unsigned __int8 *v9; // rax
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  const char *v14; // rax
  const char *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r13
  unsigned __int8 *v20; // rax
  const char *v21[4]; // [rsp-58h] [rbp-58h] BYREF
  __int16 v22; // [rsp-38h] [rbp-38h]

  if ( *(_QWORD *)(a1 + 32) )
  {
    v3 = *(_QWORD *)(a1 + 8);
    v4 = *(_QWORD *)(v3 + 232);
    if ( *(_BYTE *)(a1 + 26) )
    {
      v5 = *(_QWORD *)v4;
      if ( (*(_BYTE *)(*(_QWORD *)v4 + 2LL) & 8) == 0 )
        goto LABEL_4;
    }
    else
    {
      if ( !*(_BYTE *)(a1 + 24) )
      {
LABEL_9:
        *(_QWORD *)(a1 + 32) = 0;
        return;
      }
      v5 = *(_QWORD *)v4;
      if ( (*(_BYTE *)(*(_QWORD *)v4 + 2LL) & 8) == 0 )
        goto LABEL_12;
    }
    v8 = (unsigned __int8 *)sub_B2E500(v5);
    v9 = sub_BD3990(v8, a2);
    v10 = sub_B2A630((__int64)v9);
    if ( v10 == 9 )
    {
      if ( *(_BYTE *)(a1 + 24) )
      {
        v3 = *(_QWORD *)(a1 + 8);
        if ( !*(_BYTE *)(*(_QWORD *)(a1 + 32) + 236LL) )
        {
          (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(v3 + 224) + 1176LL))(*(_QWORD *)(v3 + 224), 0);
          v14 = sub_BD5D20(v5);
          if ( v15 && *v14 == 1 )
          {
            --v15;
            ++v14;
          }
          v16 = *(_QWORD *)(a1 + 8);
          v21[3] = v15;
          v21[2] = v14;
          v17 = *(_QWORD *)(v16 + 216);
          v22 = 1283;
          v21[0] = "$cppxdata$";
          v18 = sub_E6C460(v17, v21);
          v19 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
          v20 = (unsigned __int8 *)sub_3258F50(a1, v18);
          sub_E9A5B0(v19, v20);
          goto LABEL_6;
        }
        goto LABEL_12;
      }
      goto LABEL_5;
    }
    if ( v10 == 8 && *(_BYTE *)(v4 + 580) && !*(_BYTE *)(*(_QWORD *)(a1 + 32) + 235LL) )
    {
      (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 1176LL))(
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
        0);
      sub_3259570(a1, (__int64 *)v4, v11, v12, v13);
      goto LABEL_6;
    }
LABEL_4:
    if ( *(_BYTE *)(a1 + 24) )
    {
LABEL_13:
      v3 = *(_QWORD *)(a1 + 8);
LABEL_12:
      (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(v3 + 224) + 1176LL))(*(_QWORD *)(v3 + 224), 0);
      goto LABEL_6;
    }
LABEL_5:
    if ( !*(_BYTE *)(a1 + 25) )
    {
LABEL_6:
      v6 = *(_BYTE **)(v4 + 416);
      v7 = *(_QWORD **)(v4 + 408);
      if ( v6 != (_BYTE *)v7 )
        sub_3258570(a1 + 48, *(_BYTE **)(a1 + 56), v7, v6);
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 176LL))(
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
        *(_QWORD *)(a1 + 40),
        0);
      (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 1064LL))(
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
        0);
      goto LABEL_9;
    }
    goto LABEL_13;
  }
}
