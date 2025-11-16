// Function: sub_AF4D30
// Address: 0xaf4d30
//
__int64 __fastcall sub_AF4D30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned __int64 a9,
        __int64 a10,
        _QWORD *a11)
{
  unsigned int v11; // r10d
  __int64 v16; // rax
  unsigned __int8 v17; // dl
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // rax
  bool v22; // sf
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned int v28; // esi
  unsigned __int64 v29; // rax

  v11 = 0;
  if ( a8 )
  {
    v16 = sub_BD58A0(a2, a5, a1);
    v11 = v17;
    if ( v17 )
    {
      v18 = a7 + a6;
      v19 = a3 + 8 * v16;
      v20 = v18 - v19;
      v21 = v19 - v18;
      *a11 = v20;
      if ( a4 + v21 < 0 )
      {
        if ( *(_BYTE *)(a10 + 16) )
        {
          v11 = *(unsigned __int8 *)(a10 + 16);
          *(_QWORD *)a10 = 0;
          *(_QWORD *)(a10 + 8) = 0;
        }
        else
        {
          *(_QWORD *)a10 = 0;
          *(_QWORD *)(a10 + 8) = 0;
          *(_BYTE *)(a10 + 16) = 1;
        }
      }
      else
      {
        v22 = (__int64)(a9 + v21) < 0;
        v23 = a9 + v21;
        v24 = 0;
        if ( !v22 )
          v24 = v23;
        v25 = v24;
        if ( a9 >= v24 )
          v25 = a9;
        v26 = a4 + v23 - v24;
        if ( v26 < 0 )
          v26 = 0;
        v27 = v24 + v26;
        if ( v27 > a8 + a9 )
          v27 = a8 + a9;
        v28 = *(unsigned __int8 *)(a10 + 16);
        if ( v25 >= v27 )
        {
          v29 = 0;
          v25 = 0;
LABEL_16:
          *(_QWORD *)a10 = v29;
          *(_QWORD *)(a10 + 8) = v25;
          if ( (_BYTE)v28 )
            return v28;
          else
            *(_BYTE *)(a10 + 16) = 1;
          return v11;
        }
        v29 = v27 - v25;
        if ( a8 != v29 || a9 < v24 )
          goto LABEL_16;
        if ( (_BYTE)v28 )
        {
          v11 = *(unsigned __int8 *)(a10 + 16);
          *(_BYTE *)(a10 + 16) = 0;
        }
      }
    }
  }
  return v11;
}
