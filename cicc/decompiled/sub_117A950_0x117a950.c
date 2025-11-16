// Function: sub_117A950
// Address: 0x117a950
//
__int64 __fastcall sub_117A950(__int64 a1, const __m128i *a2, __int64 *a3)
{
  _BYTE *v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v7; // rdi
  __int64 v9; // r15
  int v10; // eax
  __int64 v11; // r8
  __int64 v12; // rax
  unsigned __int8 v13; // al
  unsigned int v14; // r12d
  bool v15; // al
  __int64 v16; // r12
  unsigned int v17; // ebx
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int8 *v20; // rdx
  unsigned __int8 *v21; // r13
  __int64 v22; // r12
  __int64 v23; // rdx
  _BYTE *v24; // rax
  unsigned int v25; // ebx
  __int64 v26; // rdx
  _BYTE *v27; // rax
  int v28; // r12d
  bool v29; // r13
  unsigned int v30; // r15d
  __int64 v31; // rax
  unsigned int v32; // r13d
  __int64 v33; // rdx
  _BYTE *v34; // rax
  __m128i v35; // xmm6
  __m128i v36; // xmm2
  unsigned __int64 v37; // xmm4_8
  __int64 v38; // rax
  char v39; // [rsp+1Fh] [rbp-B1h] BYREF
  __int64 v40; // [rsp+20h] [rbp-B0h] BYREF
  unsigned __int8 *v41; // [rsp+28h] [rbp-A8h] BYREF
  __int64 *v42[4]; // [rsp+30h] [rbp-A0h] BYREF
  _OWORD v43[4]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v44; // [rsp+90h] [rbp-40h]

  v3 = *(_BYTE **)(a1 - 96);
  v4 = *(_QWORD *)(a1 - 64);
  v39 = 0;
  v5 = *(_QWORD *)(a1 - 32);
  if ( *v3 != 82 || !*((_QWORD *)v3 - 8) )
    return 0;
  v41 = (unsigned __int8 *)*((_QWORD *)v3 - 8);
  v7 = *((_QWORD *)v3 - 4);
  v9 = v7 + 24;
  if ( *(_BYTE *)v7 != 17 )
  {
    v26 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
    if ( (unsigned int)v26 > 1 )
      return 0;
    if ( *(_BYTE *)v7 > 0x15u )
      return 0;
    v27 = sub_AD7630(v7, 0, v26);
    if ( !v27 || *v27 != 17 )
      return 0;
    v9 = (__int64)(v27 + 24);
  }
  v10 = sub_B53900((__int64)v3);
  if ( !sub_9893F0(v10, v9, &v39) )
    return 0;
  if ( !v39 )
  {
    v12 = v4;
    v4 = v5;
    v5 = v12;
  }
  v42[0] = a3;
  v42[1] = (__int64 *)&v41;
  v42[2] = &v40;
  v13 = *(_BYTE *)v4;
  if ( *(_BYTE *)v4 != 42 )
    goto LABEL_9;
  v20 = *(unsigned __int8 **)(v4 - 64);
  v21 = *(unsigned __int8 **)(v4 - 32);
  if ( v41 != v20 || !v21 )
  {
    if ( v21 != v41 || !v20 )
      return 0;
    v21 = *(unsigned __int8 **)(v4 - 64);
  }
  if ( *v41 != 52 || !*((_QWORD *)v41 - 8) )
    return 0;
  v40 = *((_QWORD *)v41 - 8);
  if ( *((unsigned __int8 **)v41 - 4) != v21 )
    goto LABEL_43;
  v35 = _mm_loadu_si128(a2 + 9);
  v36 = _mm_loadu_si128(a2 + 7);
  v37 = _mm_loadu_si128(a2 + 8).m128i_u64[0];
  v38 = a2[10].m128i_i64[0];
  v43[0] = _mm_loadu_si128(a2 + 6);
  v43[2] = v37;
  v44 = v38;
  v43[1] = v36;
  v43[3] = v35;
  if ( !(unsigned __int8)sub_9A1DB0(v21, 1, 0, (__int64)v43, v11) || v41 != (unsigned __int8 *)v5 )
  {
LABEL_43:
    v13 = *(_BYTE *)v4;
LABEL_9:
    if ( v13 == 17 )
    {
      v14 = *(_DWORD *)(v4 + 32);
      if ( v14 <= 0x40 )
        v15 = *(_QWORD *)(v4 + 24) == 1;
      else
        v15 = v14 - 1 == (unsigned int)sub_C444A0(v4 + 24);
    }
    else
    {
      v22 = *(_QWORD *)(v4 + 8);
      v23 = (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17;
      if ( (unsigned int)v23 > 1 || v13 > 0x15u )
        return 0;
      v24 = sub_AD7630(v4, 0, v23);
      if ( !v24 || *v24 != 17 )
      {
        if ( *(_BYTE *)(v22 + 8) == 17 )
        {
          v28 = *(_DWORD *)(v22 + 32);
          if ( v28 )
          {
            v29 = 0;
            v30 = 0;
            while ( 1 )
            {
              v31 = sub_AD69F0((unsigned __int8 *)v4, v30);
              if ( !v31 )
                break;
              if ( *(_BYTE *)v31 != 13 )
              {
                if ( *(_BYTE *)v31 != 17 )
                  break;
                v32 = *(_DWORD *)(v31 + 32);
                v29 = v32 <= 0x40 ? *(_QWORD *)(v31 + 24) == 1 : v32 - 1 == (unsigned int)sub_C444A0(v31 + 24);
                if ( !v29 )
                  break;
              }
              if ( v28 == ++v30 )
              {
                if ( v29 )
                  goto LABEL_13;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v25 = *((_DWORD *)v24 + 8);
      if ( v25 <= 0x40 )
        v15 = *((_QWORD *)v24 + 3) == 1;
      else
        v15 = v25 - 1 == (unsigned int)sub_C444A0((__int64)(v24 + 24));
    }
    if ( v15 )
    {
LABEL_13:
      if ( *v41 != 52 || !*((_QWORD *)v41 - 8) )
        return 0;
      v40 = *((_QWORD *)v41 - 8);
      v16 = *((_QWORD *)v41 - 4);
      if ( !v16 )
        BUG();
      if ( *(_BYTE *)v16 != 17 )
      {
        v33 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v16 + 8) + 8LL) - 17;
        if ( (unsigned int)v33 > 1 )
          return 0;
        if ( *(_BYTE *)v16 > 0x15u )
          return 0;
        v34 = sub_AD7630(v16, 0, v33);
        v16 = (__int64)v34;
        if ( !v34 || *v34 != 17 )
          return 0;
      }
      v17 = *(_DWORD *)(v16 + 32);
      if ( v17 <= 0x40 )
      {
        v18 = *(_QWORD *)(v16 + 24);
      }
      else
      {
        if ( v17 - (unsigned int)sub_C444A0(v16 + 24) > 0x40 )
          return 0;
        v18 = **(_QWORD **)(v16 + 24);
      }
      if ( v18 == 2 && v41 == (unsigned __int8 *)v5 )
      {
        v19 = sub_AD64C0(*(_QWORD *)(v5 + 8), 2, 0);
        return sub_1178AC0(v42, v19);
      }
    }
    return 0;
  }
  return sub_1178AC0(v42, (__int64)v21);
}
