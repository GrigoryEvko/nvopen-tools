// Function: sub_AC0280
// Address: 0xac0280
//
__int64 __fastcall sub_AC0280(__int64 a1, __int64 *a2, unsigned __int64 a3)
{
  _BYTE *v3; // r15
  unsigned __int64 *v6; // rsi
  __int64 v9; // rbx
  _BYTE *v10; // r13
  unsigned __int64 v11; // rdi
  unsigned int v12; // eax
  __int64 v13; // rbx
  unsigned __int64 i; // r14
  unsigned int v15; // eax
  unsigned __int64 v16; // rbx
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned __int64 v19; // r14
  __int64 v20; // r14
  unsigned __int64 v21; // rbx
  __int64 v22; // rdi
  unsigned __int64 v23; // r15
  unsigned __int64 *v24; // [rsp+0h] [rbp-90h]
  _BYTE *v25; // [rsp+8h] [rbp-88h]
  _BYTE *v26; // [rsp+10h] [rbp-80h] BYREF
  __int64 v27; // [rsp+18h] [rbp-78h]
  _BYTE v28[112]; // [rsp+20h] [rbp-70h] BYREF

  v3 = a2;
  v6 = (unsigned __int64 *)a3;
  if ( !(unsigned __int8)sub_ABEE90(a2, a3) )
  {
    *(_BYTE *)(a1 + 80) = 0;
    return a1;
  }
  v9 = 32 * a3;
  v25 = v28;
  v10 = &v3[v9];
  v26 = v28;
  v27 = 0x200000000LL;
  if ( &v3[v9] != v3 )
  {
    v11 = (unsigned __int64)v25;
    v12 = 0;
    v13 = 0;
    v24 = (unsigned __int64 *)&v26;
    for ( i = (unsigned __int64)(v3 + 32); ; i += 32LL )
    {
      v16 = v11 + 32 * v13;
      if ( !v16 )
        goto LABEL_8;
      v17 = *((_DWORD *)v3 + 2);
      *(_DWORD *)(v16 + 8) = v17;
      if ( v17 <= 0x40 )
      {
        *(_QWORD *)v16 = *(_QWORD *)v3;
        v15 = *((_DWORD *)v3 + 6);
        *(_DWORD *)(v16 + 24) = v15;
        if ( v15 <= 0x40 )
          goto LABEL_7;
      }
      else
      {
        v6 = (unsigned __int64 *)v3;
        sub_C43780(v16, v3);
        v18 = *((_DWORD *)v3 + 6);
        *(_DWORD *)(v16 + 24) = v18;
        if ( v18 <= 0x40 )
        {
LABEL_7:
          *(_QWORD *)(v16 + 16) = *((_QWORD *)v3 + 2);
          v12 = v27;
LABEL_8:
          ++v12;
          v3 = (_BYTE *)i;
          LODWORD(v27) = v12;
          if ( v10 == (_BYTE *)i )
            goto LABEL_15;
          goto LABEL_9;
        }
      }
      v6 = (unsigned __int64 *)(v3 + 16);
      v3 = (_BYTE *)i;
      sub_C43780(v16 + 16, v6);
      v12 = v27 + 1;
      LODWORD(v27) = v27 + 1;
      if ( v10 == (_BYTE *)i )
      {
LABEL_15:
        *(_QWORD *)a1 = a1 + 16;
        *(_QWORD *)(a1 + 8) = 0x200000000LL;
        if ( !v12 )
          goto LABEL_16;
        v6 = (unsigned __int64 *)&v26;
        sub_ABF400(a1, (unsigned __int64 *)&v26);
        v20 = (unsigned int)v27;
        v21 = (unsigned __int64)v26;
        *(_BYTE *)(a1 + 80) = 1;
        v19 = v21 + 32 * v20;
        if ( v21 != v19 )
        {
          do
          {
            v19 -= 32LL;
            if ( *(_DWORD *)(v19 + 24) > 0x40u )
            {
              v22 = *(_QWORD *)(v19 + 16);
              if ( v22 )
                j_j___libc_free_0_0(v22);
            }
            if ( *(_DWORD *)(v19 + 8) > 0x40u && *(_QWORD *)v19 )
              j_j___libc_free_0_0(*(_QWORD *)v19);
          }
          while ( v21 != v19 );
          v19 = (unsigned __int64)v26;
        }
        goto LABEL_26;
      }
LABEL_9:
      v13 = v12;
      v11 = (unsigned __int64)v26;
      v6 = (unsigned __int64 *)(v12 + 1LL);
      if ( (unsigned __int64)v6 > HIDWORD(v27) )
      {
        if ( (unsigned __int64)v26 > i || (unsigned __int64)&v26[32 * v12] <= i )
        {
          sub_9D5330((__int64)v24, (__int64)v6);
          v13 = (unsigned int)v27;
          v11 = (unsigned __int64)v26;
          v12 = v27;
        }
        else
        {
          v23 = i - (_QWORD)v26;
          sub_9D5330((__int64)v24, (__int64)v6);
          v11 = (unsigned __int64)v26;
          v13 = (unsigned int)v27;
          v3 = &v26[v23];
          v12 = v27;
        }
      }
    }
  }
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  *(_QWORD *)a1 = a1 + 16;
LABEL_16:
  *(_BYTE *)(a1 + 80) = 1;
  v19 = (unsigned __int64)v26;
LABEL_26:
  if ( (_BYTE *)v19 != v25 )
    _libc_free(v19, v6);
  return a1;
}
