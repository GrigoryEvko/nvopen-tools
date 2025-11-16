// Function: sub_D30F00
// Address: 0xd30f00
//
__int64 __fastcall sub_D30F00(
        unsigned __int8 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  __int64 v10; // r8
  unsigned int v11; // eax
  unsigned int v12; // r10d
  char v13; // al
  _QWORD *v15; // r15
  _QWORD *v16; // rbx
  unsigned __int8 *v17; // r13
  unsigned __int16 v18; // cx
  unsigned __int8 *v19; // r10
  __int64 v20; // r14
  unsigned __int64 v21; // rax
  unsigned __int8 *v22; // rax
  unsigned __int8 v23; // al
  unsigned __int64 v24; // r8
  unsigned __int64 v25; // r12
  char v26; // al
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int8 v30; // [rsp+0h] [rbp-60h]
  unsigned __int8 *v31; // [rsp+0h] [rbp-60h]
  unsigned __int8 v32; // [rsp+8h] [rbp-58h]
  unsigned __int64 v33; // [rsp+8h] [rbp-58h]
  unsigned __int8 v35; // [rsp+1Fh] [rbp-41h]

  v10 = 0;
  if ( a7 )
    v10 = a5;
  v35 = a2;
  LOBYTE(v11) = sub_D30550(a1, a2, (unsigned __int64 *)a3, a4, v10, a6, a7, a8);
  if ( !(_BYTE)v11 )
  {
    if ( !a5 || *(_DWORD *)(a3 + 8) > 0x40u )
      return 0;
LABEL_11:
    v15 = (_QWORD *)(a5 + 24);
    v33 = *(_QWORD *)a3;
    v16 = *(_QWORD **)(*(_QWORD *)(a5 + 40) + 56LL);
    v17 = sub_BD3990(a1, a2);
    if ( v15 == v16 )
      return 0;
    while ( 1 )
    {
      v24 = *v15 & 0xFFFFFFFFFFFFFFF8LL;
      v25 = v24;
      v15 = (_QWORD *)v24;
      if ( !v24 )
        BUG();
      v26 = *(_BYTE *)(v24 - 24);
      if ( v26 == 85 )
      {
        if ( (unsigned __int8)sub_B46490(v24 - 24) )
        {
          v27 = *(_QWORD *)(v25 - 56);
          if ( !v27 )
            return 0;
          if ( *(_BYTE *)v27
            || *(_QWORD *)(v27 + 24) != *(_QWORD *)(v25 + 56)
            || (*(_BYTE *)(v27 + 33) & 0x20) == 0
            || (unsigned int)(*(_DWORD *)(v27 + 36) - 210) > 1 )
          {
            if ( *(_BYTE *)v27 || *(_QWORD *)(v27 + 24) != *(_QWORD *)(v25 + 56) || (*(_BYTE *)(v27 + 33) & 0x20) == 0 )
              return 0;
            if ( (unsigned int)(*(_DWORD *)(v27 + 36) - 68) > 3 )
              return 0;
          }
        }
      }
      else
      {
        if ( v26 == 61 )
        {
          v18 = *(_WORD *)(v24 - 22);
          if ( (v18 & 1) != 0 )
            goto LABEL_19;
          v19 = *(unsigned __int8 **)(v24 - 56);
          v20 = *(_QWORD *)(v24 - 16);
        }
        else
        {
          if ( v26 != 62 )
            goto LABEL_19;
          v18 = *(_WORD *)(v24 - 22);
          if ( (v18 & 1) != 0 )
            goto LABEL_19;
          v19 = *(unsigned __int8 **)(v24 - 56);
          v20 = *(_QWORD *)(*(_QWORD *)(v24 - 88) + 8LL);
        }
        _BitScanReverse64(&v21, 1LL << (v18 >> 1));
        if ( (unsigned __int8)(63 - (v21 ^ 0x3F)) >= v35 )
        {
          if ( v19 == v17 )
          {
            a2 = v20;
            v31 = v19;
            v29 = sub_9208B0(a4, v20);
            v19 = v31;
            if ( v33 <= (unsigned __int64)(v29 + 7) >> 3 )
              return 1;
          }
          v22 = sub_BD3990(v19, a2);
          a2 = (__int64)v17;
          v23 = sub_D2F730(v22, v17);
          if ( v23 )
          {
            a2 = v20;
            v30 = v23;
            v28 = sub_9208B0(a4, v20);
            v12 = v30;
            if ( (unsigned __int64)(v28 + 7) >> 3 >= v33 )
              return v12;
          }
        }
      }
LABEL_19:
      if ( (_QWORD *)v25 == v16 )
        return 0;
    }
  }
  v12 = v11;
  if ( a5 )
  {
    v32 = v11;
    v13 = sub_D2F6D0(a5);
    v12 = v32;
    if ( v13 )
    {
      if ( *(_DWORD *)(a3 + 8) > 0x40u )
        return 0;
      goto LABEL_11;
    }
  }
  return v12;
}
