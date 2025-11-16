// Function: sub_5D2670
// Address: 0x5d2670
//
void __fastcall sub_5D2670(_QWORD *a1, __int64 a2, __int64 a3, int a4)
{
  __int64 *v4; // r15
  __int64 v8; // rbx
  int v9; // esi
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rcx
  _QWORD *v13; // rbx
  __int64 v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdi
  int v19; // eax
  __int64 v20; // [rsp+0h] [rbp-60h]
  _QWORD *v21; // [rsp+0h] [rbp-60h]
  __int64 v22; // [rsp+0h] [rbp-60h]
  _QWORD *v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h] BYREF
  __int64 v26; // [rsp+18h] [rbp-48h] BYREF
  __int64 v27; // [rsp+20h] [rbp-40h] BYREF
  __int64 v28; // [rsp+28h] [rbp-38h]

  v4 = (__int64 *)(a2 + 128);
  v8 = *(_QWORD *)(a2 + 128);
  v9 = 0;
  if ( v8 )
  {
    do
    {
      while ( *(_BYTE *)(v8 + 8) != 3 || v9 )
      {
        v4 = (__int64 *)*v4;
        v8 = *v4;
        if ( !*v4 )
          goto LABEL_11;
      }
      v10 = sub_736C60(3, a1);
      v25 = v8;
      v26 = v10;
      if ( a4 )
      {
        if ( !v10 )
        {
          v27 = 0;
          v17 = a3;
          v18 = 2898;
          v28 = 0;
LABEL_30:
          v22 = sub_67D9D0(v18, v17);
          sub_6855B0(2818, v25 + 56, &v27);
          sub_67E370(v22, &v27);
          sub_685910(v22);
          v9 = 1;
          goto LABEL_10;
        }
        v20 = sub_5D2590(&v25);
        v11 = sub_5D2590(&v26);
        if ( !(unsigned int)sub_5CB890(v25, v26, 0, v12) )
        {
          if ( !(unsigned int)sub_5D19D0(v25) )
          {
            v19 = sub_5D19D0(v26);
            if ( v20 != v11 && !v19 )
            {
              v27 = 0;
              v18 = 2897;
              v28 = 0;
              v17 = v26 + 56;
              goto LABEL_30;
            }
          }
          v16 = v25;
          do
          {
            if ( *(_BYTE *)(v16 + 8) == 3 )
              *(_BYTE *)(v16 + 8) = 0;
            v16 = *(_QWORD *)v16;
            v25 = v16;
          }
          while ( v16 );
        }
      }
      v9 = 1;
LABEL_10:
      v4 = (__int64 *)*v4;
      v8 = *v4;
    }
    while ( *v4 );
  }
LABEL_11:
  v24 = 0;
  v21 = 0;
  if ( a1 )
  {
    do
    {
      v13 = a1;
      a1 = (_QWORD *)*a1;
      *v13 = 0;
      v14 = sub_736C60(*((unsigned __int8 *)v13 + 8), *(_QWORD *)(a2 + 128));
      switch ( *((_BYTE *)v13 + 8) )
      {
        case 0:
          break;
        case 5:
        case 0xA:
        case 0xF:
        case 0x10:
        case 0x11:
        case 0x12:
          goto LABEL_19;
        default:
          if ( !v14 )
          {
LABEL_19:
            if ( v24 )
            {
              v15 = v21;
              v21 = v13;
              *v15 = v13;
            }
            else
            {
              v24 = v13;
              v21 = v13;
            }
          }
          break;
      }
    }
    while ( a1 );
    if ( v24 )
    {
      if ( *v4 )
        *(_QWORD *)*v4 = v24;
      else
        *v4 = (__int64)v24;
    }
  }
}
