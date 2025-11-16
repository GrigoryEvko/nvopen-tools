// Function: sub_11DAA90
// Address: 0x11daa90
//
char __fastcall sub_11DAA90(__int64 a1, int *a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  _QWORD *v8; // rcx
  _BYTE *v9; // rax
  char v10; // al
  __int64 v11; // rcx
  unsigned __int8 *v12; // rdi
  __int64 v13; // rdx
  unsigned __int8 *v14; // r15
  __int64 v15; // rbx
  __int64 v16; // rdi
  _BYTE *v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // rdx
  __m128i v21; // [rsp+0h] [rbp-80h] BYREF
  __int64 v22; // [rsp+10h] [rbp-70h]
  __int64 v23; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+28h] [rbp-58h]
  __int64 v26; // [rsp+30h] [rbp-50h]
  __int64 v27; // [rsp+38h] [rbp-48h]
  __int16 v28; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)a4 == 17 )
  {
    sub_11DA4B0(a1, a2, a3);
    if ( *(_DWORD *)(a4 + 32) <= 0x40u )
      v8 = *(_QWORD **)(a4 + 24);
    else
      v8 = **(_QWORD ***)(a4 + 24);
LABEL_4:
    LOBYTE(v9) = sub_11DA2E0(a1, (unsigned int *)a2, a3, (unsigned __int64)v8);
    return (char)v9;
  }
  v21 = (__m128i)a5;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 257;
  LOBYTE(v9) = sub_9B6260(a4, &v21, 0);
  if ( (_BYTE)v9 )
  {
    LOBYTE(v9) = sub_11DA4B0(a1, a2, a3);
    if ( *(_BYTE *)a4 == 86 )
    {
      v10 = *(_BYTE *)(a4 + 7) & 0x40;
      if ( v10 )
        v11 = *(_QWORD *)(a4 - 8);
      else
        v11 = a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF);
      v12 = *(unsigned __int8 **)(v11 + 32);
      v13 = *v12;
      v14 = v12 + 24;
      if ( (_BYTE)v13 != 17 )
      {
        LODWORD(v9) = *(unsigned __int8 *)(*((_QWORD *)v12 + 1) + 8LL) - 17;
        if ( (unsigned int)v9 > 1 )
          return (char)v9;
        if ( (unsigned __int8)v13 > 0x15u )
          return (char)v9;
        v9 = sub_AD7630((__int64)v12, 0, v13);
        if ( !v9 || *v9 != 17 )
          return (char)v9;
        v14 = v9 + 24;
        v10 = *(_BYTE *)(a4 + 7) & 0x40;
      }
      if ( v10 )
        v15 = *(_QWORD *)(a4 - 8);
      else
        v15 = a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF);
      v16 = *(_QWORD *)(v15 + 64);
      LOBYTE(v9) = *(_BYTE *)v16;
      if ( *(_BYTE *)v16 == 17 )
      {
        v17 = (_BYTE *)(v16 + 24);
LABEL_16:
        v8 = *(_QWORD **)v17;
        if ( *((_DWORD *)v17 + 2) > 0x40u )
          v8 = (_QWORD *)*v8;
        v18 = *(_QWORD **)v14;
        if ( *((_DWORD *)v14 + 2) > 0x40u )
          v18 = (_QWORD *)*v18;
        if ( v18 <= v8 )
          v8 = v18;
        goto LABEL_4;
      }
      v19 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v16 + 8) + 8LL) - 17;
      if ( (unsigned int)v19 <= 1 && (unsigned __int8)v9 <= 0x15u )
      {
        v9 = sub_AD7630(v16, 0, v19);
        if ( v9 )
        {
          if ( *v9 == 17 )
          {
            v17 = v9 + 24;
            goto LABEL_16;
          }
        }
      }
    }
  }
  return (char)v9;
}
