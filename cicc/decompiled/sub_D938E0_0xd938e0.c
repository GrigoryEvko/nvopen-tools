// Function: sub_D938E0
// Address: 0xd938e0
//
__int64 __fastcall sub_D938E0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rbp
  _BYTE *v4; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  _QWORD *v8; // rax
  _QWORD *v9; // r14
  _QWORD *v10; // rbx
  __int64 v11; // rax
  unsigned __int8 *v12; // r13
  _QWORD *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r10
  __int64 v16; // [rsp-B0h] [rbp-B0h]
  __int64 v17; // [rsp-B0h] [rbp-B0h]
  _BYTE *v18; // [rsp-A0h] [rbp-A0h] BYREF
  const void *v19; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v20; // [rsp-90h] [rbp-90h]
  const void *v21; // [rsp-88h] [rbp-88h] BYREF
  unsigned int v22; // [rsp-80h] [rbp-80h]
  char v23; // [rsp-78h] [rbp-78h]
  const void *v24; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v25; // [rsp-60h] [rbp-60h]
  const void *v26; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v27; // [rsp-50h] [rbp-50h]
  char v28; // [rsp-48h] [rbp-48h]
  __int64 v29; // [rsp-30h] [rbp-30h]
  __int64 v30; // [rsp-8h] [rbp-8h]

  v30 = v2;
  v29 = v1;
  switch ( *(_WORD *)(a1 + 24) )
  {
    case 0:
      return *(_QWORD *)(a1 + 32);
    case 1:
    case 3:
    case 4:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
    case 0x10:
      return 0;
    case 2:
      v7 = sub_D938E0(*(_QWORD *)(a1 + 32));
      if ( !v7 )
        return 0;
      return sub_AD4C30(v7, *(__int64 ***)(a1 + 40), 0);
    case 5:
      v8 = *(_QWORD **)(a1 + 32);
      v9 = &v8[*(_QWORD *)(a1 + 40)];
      if ( v9 == v8 )
        return 0;
      v10 = *(_QWORD **)(a1 + 32);
      v4 = 0;
      break;
    case 0xE:
      v6 = sub_D938E0(*(_QWORD *)(a1 + 32));
      if ( !v6 )
        return 0;
      return sub_AD4C50(v6, *(__int64 ***)(a1 + 40), 0);
    case 0xF:
      v4 = *(_BYTE **)(a1 - 8);
      if ( *v4 > 0x15u )
        return 0;
      return (__int64)v4;
    default:
      BUG();
  }
  while ( 1 )
  {
    v11 = sub_D938E0(*v10);
    v12 = (unsigned __int8 *)v11;
    if ( !v11 )
      break;
    if ( v4 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(v11 + 8) + 8LL) == 14 )
      {
        v23 = 0;
        v13 = (_QWORD *)sub_BD5C60((__int64)v4);
        v14 = sub_BCB2B0(v13);
        v28 = 0;
        v15 = v14;
        if ( v23 )
        {
          v25 = v20;
          if ( v20 > 0x40 )
          {
            v16 = v14;
            sub_C43780((__int64)&v24, &v19);
            v15 = v16;
          }
          else
          {
            v24 = v19;
          }
          v27 = v22;
          if ( v22 > 0x40 )
          {
            v17 = v15;
            sub_C43780((__int64)&v26, &v21);
            v15 = v17;
          }
          else
          {
            v26 = v21;
          }
          v28 = 1;
        }
        v18 = v4;
        v4 = (_BYTE *)sub_AD9FD0(v15, v12, (__int64 *)&v18, 1, 0, (__int64)&v24, 0);
        if ( v28 )
        {
          v28 = 0;
          if ( v27 > 0x40 && v26 )
            j_j___libc_free_0_0(v26);
          if ( v25 > 0x40 && v24 )
            j_j___libc_free_0_0(v24);
        }
        if ( v23 )
        {
          v23 = 0;
          if ( v22 > 0x40 && v21 )
            j_j___libc_free_0_0(v21);
          if ( v20 > 0x40 )
          {
            if ( v19 )
              j_j___libc_free_0_0(v19);
          }
        }
      }
      else
      {
        v4 = (_BYTE *)sub_AD57C0((__int64)v4, (unsigned __int8 *)v11, 0, 0);
      }
    }
    else
    {
      v4 = (_BYTE *)v11;
    }
    if ( v9 == ++v10 )
      return (__int64)v4;
  }
  return 0;
}
