// Function: sub_36E1BC0
// Address: 0x36e1bc0
//
unsigned __int64 __fastcall sub_36E1BC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r13
  __int64 v7; // rdi
  char v8; // bl
  unsigned int v9; // eax
  unsigned __int8 v10; // bl
  unsigned int v11; // edx
  bool v12; // cl
  bool v13; // di
  unsigned __int8 v14; // r13
  __int64 v15; // rbx
  __int16 v17; // ax
  unsigned int v18; // eax
  int v19; // eax
  __int64 v20; // r9
  __int64 v21; // rax
  const char *v22; // rdx
  char *v23; // rdx
  __m128i v25[2]; // [rsp+10h] [rbp-E0h] BYREF
  _QWORD v26[4]; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v27; // [rsp+50h] [rbp-A0h]
  __m128i v28; // [rsp+60h] [rbp-90h] BYREF
  _QWORD **v29; // [rsp+70h] [rbp-80h]
  __int64 v30; // [rsp+78h] [rbp-78h]
  char v31; // [rsp+80h] [rbp-70h]
  _QWORD v32[2]; // [rsp+88h] [rbp-68h] BYREF
  _QWORD *v33; // [rsp+98h] [rbp-58h] BYREF

  v6 = *(_QWORD *)(a1 + 1136);
  v7 = *(_QWORD *)(a4 + 112);
  v8 = *(_BYTE *)(v7 + 37);
  v9 = sub_36D7800(v7);
  v10 = v8 & 0xF;
  if ( *(_DWORD *)(v6 + 344) <= 0x45u )
  {
    v12 = 0;
    v13 = 0;
  }
  else
  {
    v11 = *(_DWORD *)(v6 + 336);
    v12 = v11 > 0x3B;
    v13 = v11 > 0x51;
  }
  if ( v9 - 4 <= 1 || v9 == 101 )
  {
    v14 = 0;
LABEL_9:
    v15 = 0;
    return v14 | (unsigned __int64)(v15 << 32);
  }
  if ( v10 > 2u && !v12 )
  {
    v28.m128i_i64[1] = 124;
    v23 = (&off_4B91120)[v10];
    v28.m128i_i64[0] = (__int64)"PTX does not support \"atomic\" for orderings different than\"NotAtomic\" or \"Monotonic"
                                "\" for sm_60 or older, but order is: \"{}\".";
LABEL_30:
    v32[1] = v23;
    v29 = &v33;
    v30 = 1;
    v31 = 1;
    v32[0] = &unk_49E6678;
    v33 = v32;
    v27 = 263;
    v26[0] = &v28;
    sub_C64D30((__int64)v26, 1u);
  }
  if ( v9 > 1 )
  {
    v14 = 0;
    if ( v9 != 3 )
      goto LABEL_9;
  }
  switch ( v10 )
  {
    case 0u:
      v14 = *(_BYTE *)(a4 + 32) & 8;
      goto LABEL_9;
    case 1u:
    case 2u:
      if ( (*(_BYTE *)(a4 + 32) & 8) == 0 )
      {
        if ( v12 )
        {
          v14 = 2;
          v15 = (unsigned int)sub_36E1AA0(a1, a4);
          return v14 | (unsigned __int64)(v15 << 32);
        }
LABEL_17:
        v14 = 8;
        goto LABEL_9;
      }
      if ( v9 != 1 )
        goto LABEL_17;
      v14 = 9;
      v15 = 4;
      if ( !v13 )
        goto LABEL_17;
      return v14 | (unsigned __int64)(v15 << 32);
    case 4u:
      if ( (*(_BYTE *)(*(_QWORD *)(a4 + 112) + 32LL) & 1) != 0 )
      {
        v14 = 4;
        v15 = (unsigned int)sub_36E1AA0(a1, a4);
        return v14 | (unsigned __int64)(v15 << 32);
      }
      sub_3418C90(v25, a4, 0);
      v22 = "PTX only supports Acquire Ordering on reads: {}";
      goto LABEL_28;
    case 5u:
      if ( (*(_BYTE *)(*(_QWORD *)(a4 + 112) + 32LL) & 2) != 0 )
      {
        v14 = 5;
        v15 = (unsigned int)sub_36E1AA0(a1, a4);
        return v14 | (unsigned __int64)(v15 << 32);
      }
      sub_3418C90(v25, a4, 0);
      v22 = "PTX only supports Release Ordering on writes: {}";
      goto LABEL_28;
    case 6u:
      sub_3418C90(v25, a4, 0);
      v22 = "NVPTX does not support AcquireRelease Ordering on read-modify-write yet and PTX does not support it on loads or stores: {}";
      goto LABEL_28;
    case 7u:
      v17 = *(_WORD *)(*(_QWORD *)(a4 + 112) + 32LL);
      if ( (v17 & 1) != 0 )
      {
        v14 = 4;
      }
      else
      {
        if ( (v17 & 2) == 0 )
        {
          sub_3418C90(v25, a4, 0);
          v22 = "NVPTX does not support SequentiallyConsistent Ordering on read-modify-writes yet: {}";
LABEL_28:
          sub_35EF270(&v28, 1, v22, v25);
          v27 = 263;
          v26[0] = &v28;
          sub_C64D30((__int64)v26, 1u);
        }
        v14 = 5;
      }
      v18 = sub_36E1AA0(a1, a4);
      v15 = v18;
      v19 = sub_36DCF60(7, v18, *(_QWORD *)(a1 + 1136));
      v21 = sub_33F7740(*(_QWORD **)(a1 + 64), v19, a2, 1u, 0, v20, *(_OWORD *)a3);
      *(_DWORD *)(a3 + 8) = 0;
      *(_QWORD *)a3 = v21;
      return v14 | (unsigned __int64)(v15 << 32);
    default:
      v28.m128i_i64[1] = 55;
      v23 = (&off_4B91120)[v10];
      v28.m128i_i64[0] = (__int64)"NVPTX backend does not support AtomicOrdering \"{}\" yet.";
      goto LABEL_30;
  }
}
