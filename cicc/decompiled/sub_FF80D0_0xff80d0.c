// Function: sub_FF80D0
// Address: 0xff80d0
//
__int64 __fastcall sub_FF80D0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rax
  unsigned int v7; // r14d
  __int64 v9; // rax
  __int64 v11; // r12
  __int16 v12; // ax
  unsigned int v13; // edi
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rcx
  _QWORD *v17; // rdi
  __int64 v18; // rax
  _DWORD *v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rdx
  _BYTE *v22; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v23; // [rsp-B0h] [rbp-B0h]
  _BYTE v24[48]; // [rsp-A8h] [rbp-A8h] BYREF
  _QWORD *v25; // [rsp-78h] [rbp-78h]
  __int64 v26; // [rsp-70h] [rbp-70h]
  _QWORD v27[13]; // [rsp-68h] [rbp-68h] BYREF

  v6 = a2[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v6 == a2 + 6 )
    goto LABEL_31;
  if ( !v6 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 > 0xA )
LABEL_31:
    BUG();
  v7 = 0;
  if ( *(_BYTE *)(v6 - 24) == 31 && (*(_DWORD *)(v6 - 20) & 0x7FFFFFF) == 3 )
  {
    v9 = *(_QWORD *)(v6 - 120);
    if ( *(_BYTE *)v9 == 83 )
    {
      v11 = (__int64)a2;
      v22 = v24;
      v23 = 0xC00000000LL;
      v12 = *(_WORD *)(v9 + 2);
      v13 = v12 & 0x3F;
      LOBYTE(v7) = (v12 & 0x37) == 6 || (v12 & 0x37) == 1;
      if ( (_BYTE)v7 )
      {
        if ( (unsigned __int8)sub_B535D0(v13) )
        {
          v14 = (unsigned __int64)(unsigned int)dword_4F8E5F4 << 32;
          v15 = (unsigned int)dword_4F8E5F0;
        }
        else
        {
          v14 = (unsigned __int64)(unsigned int)dword_4F8E5F0 << 32;
          v15 = (unsigned int)dword_4F8E5F4;
        }
        v27[0] = v14 | v15;
        v26 = 0xC00000002LL;
        v25 = v27;
        if ( (unsigned int)v23 > 1uLL )
        {
          *(_QWORD *)v22 = v27[0];
          v17 = v25;
          LODWORD(v23) = 2;
        }
        else
        {
          v16 = 4LL * (unsigned int)v23;
          v17 = v25;
          a2 = (_QWORD *)((char *)v25 + v16);
          if ( (_QWORD *)((char *)v17 + v16) != v17 + 1 )
          {
            memcpy(&v22[v16], a2, 8 - v16);
            v17 = v25;
          }
          LODWORD(v23) = 2;
        }
        if ( v17 != v27 )
          _libc_free(v17, a2);
        goto LABEL_17;
      }
      v18 = qword_4F8E5D0;
      if ( qword_4F8E5D0 )
      {
        v19 = &unk_4F8E5C8;
        do
        {
          v20 = *(_QWORD *)(v18 + 16);
          v21 = *(_QWORD *)(v18 + 24);
          if ( v13 > *(_DWORD *)(v18 + 32) )
          {
            v18 = *(_QWORD *)(v18 + 24);
          }
          else
          {
            v19 = (_DWORD *)v18;
            v18 = *(_QWORD *)(v18 + 16);
          }
        }
        while ( v18 );
        if ( v19 != (_DWORD *)&unk_4F8E5C8 && v13 >= v19[8] )
        {
          sub_FEE1E0((__int64)&v22, (__int64)(v19 + 10), v21, v20, (__int64)&unk_4F8E5C8, a6);
LABEL_17:
          v7 = 1;
          sub_FF6650(a1, v11, (__int64)&v22);
          if ( v22 != v24 )
            _libc_free(v22, v11);
        }
      }
    }
  }
  return v7;
}
