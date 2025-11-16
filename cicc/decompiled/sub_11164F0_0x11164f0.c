// Function: sub_11164F0
// Address: 0x11164f0
//
_QWORD *__fastcall sub_11164F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // r12
  __int16 v8; // bx
  __int64 v9; // r8
  _QWORD *v10; // r14
  __int64 v12; // rbx
  _QWORD **v13; // rdx
  int v14; // ecx
  int v15; // eax
  __int64 *v16; // rax
  __int64 v17; // rsi
  const void *v18; // rdi
  __int64 v19; // rdx
  _BYTE *v20; // rax
  unsigned int v21; // eax
  __int64 v22; // rbx
  _QWORD **v23; // rdx
  int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // rsi
  __int64 v27; // [rsp+8h] [rbp-A8h]
  __int64 v28; // [rsp+8h] [rbp-A8h]
  __int64 v29; // [rsp+18h] [rbp-98h]
  const void *v30; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v31; // [rsp+28h] [rbp-88h]
  __int64 v32; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-78h]
  const void *v34; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v35; // [rsp+48h] [rbp-68h]
  _BYTE v36[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v37; // [rsp+70h] [rbp-40h]

  v5 = *(_QWORD *)(a3 - 64);
  v6 = *(_QWORD *)(a3 - 32);
  v7 = *(_QWORD *)(a3 + 8);
  v8 = *(_WORD *)(a2 + 2) & 0x3F;
  v9 = v5 + 24;
  if ( *(_BYTE *)v5 != 17 )
  {
    v19 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
    if ( (unsigned int)v19 > 1 )
      return 0;
    if ( *(_BYTE *)v5 > 0x15u )
      return 0;
    v20 = sub_AD7630(v5, 0, v19);
    if ( !v20 )
      return 0;
    v9 = (__int64)(v20 + 24);
    if ( *v20 != 17 )
      return 0;
  }
  if ( v8 == 34 )
  {
    v31 = *(_DWORD *)(a4 + 8);
    if ( v31 > 0x40 )
    {
      v28 = v9;
      sub_C43780((__int64)&v30, (const void **)a4);
      v9 = v28;
    }
    else
    {
      v30 = *(const void **)a4;
    }
    v27 = v9;
    sub_C46A40((__int64)&v30, 1);
    v21 = v31;
    v31 = 0;
    v33 = v21;
    v32 = (__int64)v30;
    sub_C4A1D0((__int64)&v34, v27, (__int64)&v32);
    v22 = sub_AD8D80(v7, (__int64)&v34);
    v37 = 257;
    v10 = sub_BD2C40(72, unk_3F10FD0);
    if ( v10 )
    {
      v23 = *(_QWORD ***)(v6 + 8);
      v24 = *((unsigned __int8 *)v23 + 8);
      if ( (unsigned int)(v24 - 17) > 1 )
      {
        v26 = sub_BCB2A0(*v23);
      }
      else
      {
        BYTE4(v29) = (_BYTE)v24 == 18;
        LODWORD(v29) = *((_DWORD *)v23 + 8);
        v25 = (__int64 *)sub_BCB2A0(*v23);
        v26 = sub_BCE1B0(v25, v29);
      }
      sub_B523C0((__int64)v10, v26, 53, 37, v6, v22, (__int64)v36, 0, 0, 0);
    }
    if ( v35 > 0x40 && v34 )
      j_j___libc_free_0_0(v34);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    if ( v31 > 0x40 )
    {
      v18 = v30;
      if ( v30 )
        goto LABEL_11;
    }
  }
  else
  {
    v10 = 0;
    if ( v8 == 36 )
    {
      sub_C4A1D0((__int64)&v34, v9, a4);
      v12 = sub_AD8D80(v7, (__int64)&v34);
      v37 = 257;
      v10 = sub_BD2C40(72, unk_3F10FD0);
      if ( v10 )
      {
        v13 = *(_QWORD ***)(v6 + 8);
        v14 = *((unsigned __int8 *)v13 + 8);
        if ( (unsigned int)(v14 - 17) > 1 )
        {
          v17 = sub_BCB2A0(*v13);
        }
        else
        {
          v15 = *((_DWORD *)v13 + 8);
          BYTE4(v32) = (_BYTE)v14 == 18;
          LODWORD(v32) = v15;
          v16 = (__int64 *)sub_BCB2A0(*v13);
          v17 = sub_BCE1B0(v16, v32);
        }
        sub_B523C0((__int64)v10, v17, 53, 34, v6, v12, (__int64)v36, 0, 0, 0);
      }
      if ( v35 > 0x40 )
      {
        v18 = v34;
        if ( v34 )
LABEL_11:
          j_j___libc_free_0_0(v18);
      }
    }
  }
  return v10;
}
