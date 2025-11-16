// Function: sub_36D4590
// Address: 0x36d4590
//
void __fastcall sub_36D4590(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi
  __int64 *v4; // r14
  __int64 v6; // r15
  __int64 (*v7)(void); // rax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rbx
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rbx
  _QWORD *v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // rbx
  __int32 v19; // eax
  int v20; // eax
  __int64 v21; // rax
  __int32 v22; // ebx
  __int64 v23; // r12
  _QWORD *v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int32 v28; // [rsp+8h] [rbp-A8h]
  unsigned int v29; // [rsp+10h] [rbp-A0h]
  __int64 v31; // [rsp+20h] [rbp-90h]
  __int64 v32; // [rsp+28h] [rbp-88h] BYREF
  __int64 v33; // [rsp+30h] [rbp-80h] BYREF
  __int64 v34; // [rsp+38h] [rbp-78h]
  __int64 v35; // [rsp+40h] [rbp-70h]
  __m128i v36; // [rsp+50h] [rbp-60h] BYREF
  __int64 v37; // [rsp+60h] [rbp-50h]
  __int64 v38; // [rsp+68h] [rbp-48h]
  __int64 v39; // [rsp+70h] [rbp-40h]

  if ( *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL) != *(_QWORD *)(*(_QWORD *)(a2 + 48) + 16LL) )
  {
    v3 = *(_QWORD *)(a2 + 16);
    v4 = *(__int64 **)(a3 + 56);
    v6 = *(_QWORD *)(a2 + 32);
    v7 = *(__int64 (**)(void))(*(_QWORD *)v3 + 200LL);
    if ( (char *)v7 == (char *)sub_3020000 )
      v8 = v3 + 456;
    else
      v8 = v7();
    v9 = *(_QWORD *)(a2 + 8);
    v31 = 0;
    v10 = 6999 - (unsigned int)(*(_BYTE *)(v9 + 1264) == 0);
    v29 = 2912 - (*(_BYTE *)(v9 + 1264) == 0);
    v11 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v8 + 680LL))(v8, a2);
    if ( v11 < 0 )
      v12 = *(_QWORD *)(*(_QWORD *)(v6 + 56) + 16LL * (v11 & 0x7FFFFFFF) + 8);
    else
      v12 = *(_QWORD *)(*(_QWORD *)(v6 + 304) + 8LL * (unsigned int)v11);
    if ( v12 )
    {
      if ( (*(_BYTE *)(v12 + 3) & 0x10) != 0 )
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)(v12 + 32);
          if ( !v12 )
            break;
          if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
            goto LABEL_8;
        }
      }
      else
      {
LABEL_8:
        v28 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v8 + 680LL))(v8, a2);
        v13 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 128LL))(*(_QWORD *)(a2 + 16));
        v14 = 5 * v10;
        v15 = *(_QWORD *)(v13 + 8);
        v32 = v31;
        v33 = 0;
        v34 = 0;
        v35 = 0;
        v16 = sub_2F26260(a3, v4, &v33, v15 - 8 * v14, v28);
        v4 = v17;
        v18 = (__int64)v16;
        v19 = sub_30590D0(v8, a2);
        v36.m128i_i64[0] = 0;
        v37 = 0;
        v36.m128i_i32[2] = v19;
        v38 = 0;
        v39 = 0;
        sub_2E8EAD0((__int64)v4, v18, &v36);
        if ( v33 )
          sub_B91220((__int64)&v33, v33);
        if ( v32 )
          sub_B91220((__int64)&v32, v32);
      }
    }
    v20 = sub_30590D0(v8, a2);
    if ( v20 < 0 )
      v21 = *(_QWORD *)(*(_QWORD *)(v6 + 56) + 16LL * (v20 & 0x7FFFFFFF) + 8);
    else
      v21 = *(_QWORD *)(*(_QWORD *)(v6 + 304) + 8LL * (unsigned int)v20);
    if ( v21 )
    {
      if ( (*(_BYTE *)(v21 + 3) & 0x10) != 0 )
      {
        while ( 1 )
        {
          v21 = *(_QWORD *)(v21 + 32);
          if ( !v21 )
            break;
          if ( (*(_BYTE *)(v21 + 3) & 0x10) == 0 )
            goto LABEL_16;
        }
      }
      else
      {
LABEL_16:
        v22 = sub_30590D0(v8, a2);
        v23 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 128LL))(*(_QWORD *)(a2 + 16)) + 8);
        v32 = v31;
        v33 = 0;
        v34 = 0;
        v35 = 0;
        v24 = sub_2F26260(a3, v4, &v33, v23 - 40LL * v29, v22);
        v36.m128i_i64[0] = 1;
        v25 = (__int64)v24;
        v26 = *(unsigned int *)(a2 + 336);
        v37 = 0;
        v38 = v26;
        sub_2E8EAD0(v27, v25, &v36);
        if ( v33 )
          sub_B91220((__int64)&v33, v33);
        if ( v32 )
          sub_B91220((__int64)&v32, v32);
      }
    }
  }
}
