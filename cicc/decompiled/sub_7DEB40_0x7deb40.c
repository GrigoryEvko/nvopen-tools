// Function: sub_7DEB40
// Address: 0x7deb40
//
__int64 __fastcall sub_7DEB40(__int64 a1)
{
  _QWORD *v2; // rbx
  __m128i *v3; // rax
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 i; // rdx
  __int64 v7; // r14
  __int64 v8; // r13
  _BYTE *v9; // r15
  __int64 v10; // rax
  _BYTE *v11; // r15
  __int64 v12; // rsi
  _QWORD *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rbx
  _QWORD *v19; // rax
  _BYTE *v20; // rax
  __int64 v21; // rdi
  _QWORD *v22; // rdi
  _BYTE *v23; // rax
  __int64 v24; // rbx
  __int64 *v25; // r14
  _QWORD *v27; // rcx
  _QWORD *v28; // rdi
  const __m128i *v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // [rsp+0h] [rbp-D0h]
  __int64 v35; // [rsp+0h] [rbp-D0h]
  __int64 v36; // [rsp+10h] [rbp-C0h]
  _QWORD *v37; // [rsp+10h] [rbp-C0h]
  _QWORD *v38; // [rsp+10h] [rbp-C0h]
  __int64 v39; // [rsp+18h] [rbp-B8h]
  __int64 v40; // [rsp+18h] [rbp-B8h]
  __int64 v41; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v42; // [rsp+28h] [rbp-A8h] BYREF
  _BYTE v43[32]; // [rsp+30h] [rbp-A0h] BYREF
  _BYTE v44[128]; // [rsp+50h] [rbp-80h] BYREF

  v2 = *(_QWORD **)(a1 + 56);
  if ( v2 )
  {
    sub_72BA30(byte_4F06A51[0]);
    v3 = (__m128i *)sub_7E1E20(*v2);
    v4 = sub_7DB130(v3, &v42, &v41);
    v5 = sub_7DDA20(v4);
    v39 = *v2;
    sub_7EAF80(*v2);
    for ( i = v39; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v36 = i;
    v40 = v2[1];
    v7 = sub_72D2E0((_QWORD *)i);
    v8 = sub_7E7CB0(v7);
    v9 = sub_73E230(v5, 0);
    v10 = ((__int64 (*)(void))sub_7DB8E0)();
    v11 = sub_73E130(v9, v10);
    v12 = byte_4F06A51[0];
    v13 = sub_73A8E0(*(_QWORD *)(v36 + 128), byte_4F06A51[0]);
    v14 = v41;
    *((_QWORD *)v11 + 2) = v13;
    v37 = v13;
    if ( v14 )
    {
      v15 = sub_7E2510(v14);
      v16 = v37;
      v38 = 0;
      v16[2] = v15;
      v17 = v2[2];
      if ( v17 )
        goto LABEL_6;
    }
    else
    {
      v12 = unk_4F06870;
      v27 = sub_73A830(v42, unk_4F06870);
      v37[2] = v27;
      v17 = v2[2];
      v38 = v27;
      if ( v17 )
      {
LABEL_6:
        v34 = sub_7FDF40(v17, 1, 0);
        v18 = sub_7F9D60();
        v19 = sub_731330(v34);
        v20 = sub_73E110((__int64)v19, v18);
        v21 = qword_4F18810;
        v38[2] = v20;
        if ( !v21 )
        {
          v35 = sub_7F9D60();
          v31 = unk_4F06870;
          sub_72BA30(unk_4F06870);
          sub_7DB8E0(v31, v18);
          sub_7E1C10(v31, v18);
          v22 = (_QWORD *)sub_7F8AB0("__throw_setup_dtor", v35, 0, 0, 0, (__int64)v11);
          goto LABEL_8;
        }
        goto LABEL_7;
      }
    }
    if ( v41 )
    {
      v21 = qword_4F18808;
      if ( !qword_4F18808 )
      {
        v28 = sub_72BA30(unk_4F06870);
        sub_72D2E0(v28);
        sub_7DB8E0(v28, 0);
        sub_7E1C10(v28, 0);
        v22 = (_QWORD *)sub_7F8AB0("__throw_setup_ptr", 0, 0, 0, 0, (__int64)v11);
        goto LABEL_8;
      }
    }
    else
    {
      v21 = qword_4F18818;
      if ( !qword_4F18818 )
      {
        v32 = unk_4F06870;
        sub_72BA30(unk_4F06870);
        ((void (*)(void))sub_7DB8E0)();
        sub_7E1C10(v32, v12);
        v22 = (_QWORD *)sub_7F8AB0("__throw_setup", 0, 0, 0, 0, (__int64)v11);
        goto LABEL_8;
      }
    }
LABEL_7:
    v22 = (_QWORD *)sub_7F88E0(v21, v11);
LABEL_8:
    v23 = sub_73E130(v22, v7);
    v24 = sub_7E2BE0(v8, v23);
    if ( qword_4F18800 )
    {
      v25 = (__int64 *)sub_7F88E0(qword_4F18800, 0);
    }
    else
    {
      v30 = sub_72CBE0();
      v25 = (__int64 *)sub_7F8B20("__throw", &qword_4F18800, v30, 0, 0, 0);
    }
    *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(qword_4F18800 + 152) + 168LL) + 20LL) |= 1u;
    *(_QWORD *)(v24 + 16) = v25;
    sub_7264E0(a1, 1);
    sub_73D8E0(a1, 0x5Bu, *v25, 0, v24);
    sub_7F90D0(v8, v44);
    sub_7E1780(v25, v43);
    return sub_7FEC50(v40, (unsigned int)v44, 0, 0, 2, 0, (__int64)v43, 0, 0);
  }
  if ( qword_4F187F8 )
  {
    v29 = (const __m128i *)sub_7F88E0(qword_4F187F8, 0);
  }
  else
  {
    v33 = sub_72CBE0();
    v29 = (const __m128i *)sub_7F8B20("__rethrow", &qword_4F187F8, v33, 0, 0, 0);
  }
  *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(qword_4F187F8 + 152) + 168LL) + 20LL) |= 1u;
  return sub_730620(a1, v29);
}
