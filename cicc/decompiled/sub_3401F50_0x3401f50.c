// Function: sub_3401F50
// Address: 0x3401f50
//
unsigned __int8 *__fastcall sub_3401F50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __m128i a7)
{
  unsigned __int8 *result; // rax
  _DWORD *v10; // rax
  _DWORD *v11; // r15
  __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  char v14; // al
  unsigned int v15; // eax
  unsigned int v16; // ebx
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  _DWORD *v19; // rdi
  __int64 v20; // rdx
  unsigned int v21; // eax
  unsigned int v22; // ebx
  __int64 v23; // rdx
  __m128i v24; // xmm0
  _DWORD *v25; // rax
  _DWORD *v26; // r15
  __int64 v27; // rax
  __int64 v28; // [rsp+0h] [rbp-90h]
  _DWORD *v29; // [rsp+8h] [rbp-88h]
  unsigned __int8 *v30; // [rsp+8h] [rbp-88h]
  _DWORD *v31; // [rsp+8h] [rbp-88h]
  __int64 v32; // [rsp+10h] [rbp-80h] BYREF
  __int64 v33; // [rsp+18h] [rbp-78h]
  unsigned __int64 v34; // [rsp+20h] [rbp-70h] BYREF
  char v35; // [rsp+28h] [rbp-68h]
  __int64 v36; // [rsp+30h] [rbp-60h] BYREF
  __int64 v37; // [rsp+38h] [rbp-58h]
  unsigned __int64 v38; // [rsp+40h] [rbp-50h] BYREF
  __int64 v39; // [rsp+48h] [rbp-48h]

  v32 = a4;
  v33 = a5;
  if ( (unsigned int)a2 > 0xBC )
  {
    if ( (unsigned int)a2 <= 0x118 )
    {
      if ( (unsigned int)a2 <= 0x116 )
        return 0;
      v31 = sub_300AC80((unsigned __int16 *)&v32, a2);
      v25 = sub_C33340();
      v26 = v25;
      if ( (a6 & 0x20) != 0 )
      {
        if ( (a6 & 0x40) != 0 )
        {
          if ( v31 == v25 )
            sub_C3C500(&v38, (__int64)v25);
          else
            sub_C373C0(&v38, (__int64)v31);
          if ( (_DWORD *)v38 == v26 )
            sub_C3CF90((__int64)&v38, 0);
          else
            sub_C35910((__int64)&v38, 0);
        }
        else
        {
          if ( v31 == v25 )
            sub_C3C500(&v38, (__int64)v25);
          else
            sub_C373C0(&v38, (__int64)v31);
          if ( (_DWORD *)v38 == v26 )
            sub_C3CF20((__int64)&v38, 0);
          else
            sub_C36EF0((_DWORD **)&v38, 0);
        }
      }
      else
      {
        if ( v31 == v25 )
          sub_C3C500(&v38, (__int64)v25);
        else
          sub_C373C0(&v38, (__int64)v31);
        if ( (_DWORD *)v38 == v26 )
          sub_C3D480((__int64)&v38, 0, 0, 0);
        else
          sub_C36070((__int64)&v38, 0, 0, 0);
      }
      if ( (_DWORD)a2 != 280 )
        goto LABEL_16;
      if ( (_DWORD *)v38 != v26 )
        goto LABEL_74;
    }
    else
    {
      if ( (unsigned int)(a2 - 283) > 1 )
        return 0;
      v29 = sub_300AC80((unsigned __int16 *)&v32, a2);
      v10 = sub_C33340();
      v11 = v10;
      if ( (a6 & 0x40) != 0 )
      {
        if ( v29 == v10 )
          sub_C3C500(&v38, (__int64)v10);
        else
          sub_C373C0(&v38, (__int64)v29);
        if ( (_DWORD *)v38 == v11 )
          sub_C3CF90((__int64)&v38, 0);
        else
          sub_C35910((__int64)&v38, 0);
      }
      else
      {
        if ( v29 == v10 )
          sub_C3C500(&v38, (__int64)v10);
        else
          sub_C373C0(&v38, (__int64)v29);
        if ( (_DWORD *)v38 == v11 )
          sub_C3CF20((__int64)&v38, 0);
        else
          sub_C36EF0((_DWORD **)&v38, 0);
      }
      if ( (_DWORD)a2 != 284 )
      {
LABEL_16:
        v28 = sub_33FE6E0(a1, (__int64 *)&v38, a3, v32, v33, 0, a7);
        sub_91D830(&v38);
        return (unsigned __int8 *)v28;
      }
      if ( (_DWORD *)v38 != v11 )
      {
LABEL_74:
        sub_C34440((unsigned __int8 *)&v38);
        goto LABEL_16;
      }
    }
    sub_C3CCB0((__int64)&v38);
    goto LABEL_16;
  }
  if ( (unsigned int)a2 > 0xB3 )
  {
    switch ( (int)a2 )
    {
      case 180:
        if ( (_WORD)v32 )
        {
          if ( (_WORD)v32 == 1 || (unsigned __int16)(v32 - 504) <= 7u )
            BUG();
          v27 = 16LL * ((unsigned __int16)v32 - 1);
          v13 = *(_QWORD *)&byte_444C4A0[v27];
          v14 = byte_444C4A0[v27 + 8];
        }
        else
        {
          v38 = sub_3007260((__int64)&v32);
          v39 = v12;
          v13 = v38;
          v14 = v39;
        }
        v34 = v13;
        v35 = v14;
        v15 = sub_CA1930(&v34);
        LODWORD(v37) = v15;
        v16 = v15;
        if ( v15 <= 0x40 )
        {
          v17 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v15;
          if ( !v15 )
            v17 = 0;
          v36 = v17;
          v18 = ~(1LL << ((unsigned __int8)v15 - 1));
          goto LABEL_31;
        }
        sub_C43690((__int64)&v36, -1, 1);
        v18 = ~(1LL << ((unsigned __int8)v16 - 1));
        if ( (unsigned int)v37 <= 0x40 )
        {
LABEL_31:
          v36 &= v18;
          goto LABEL_32;
        }
        *(_QWORD *)(v36 + 8LL * ((v16 - 1) >> 6)) &= v18;
LABEL_32:
        result = sub_34007B0(a1, (__int64)&v36, a3, v32, v33, 0, a7, 0);
        if ( (unsigned int)v37 <= 0x40 )
          return result;
        v19 = (_DWORD *)v36;
        if ( !v36 )
          return result;
        goto LABEL_34;
      case 181:
        v36 = sub_2D5B750((unsigned __int16 *)&v32);
        v37 = v20;
        v21 = sub_CA1930(&v36);
        LODWORD(v39) = v21;
        v22 = v21;
        if ( v21 > 0x40 )
        {
          sub_C43690((__int64)&v38, 0, 0);
          v23 = 1LL << ((unsigned __int8)v22 - 1);
          if ( (unsigned int)v39 > 0x40 )
          {
            *(_QWORD *)(v38 + 8LL * ((v22 - 1) >> 6)) |= v23;
LABEL_38:
            result = sub_34007B0(a1, (__int64)&v38, a3, v32, v33, 0, a7, 0);
            if ( (unsigned int)v39 <= 0x40 )
              return result;
            v19 = (_DWORD *)v38;
            if ( !v38 )
              return result;
LABEL_34:
            v30 = result;
            j_j___libc_free_0_0((unsigned __int64)v19);
            return v30;
          }
        }
        else
        {
          v38 = 0;
          v23 = 1LL << ((unsigned __int8)v21 - 1);
        }
        v38 |= v23;
        goto LABEL_38;
      case 182:
      case 186:
        return sub_34015B0(a1, a3, (unsigned int)v32, v33, 0, 0, a7);
      case 183:
      case 187:
      case 188:
        return sub_3400BD0(a1, 0, a3, (unsigned int)v32, v33, 0, a7, 0);
      default:
        return 0;
    }
  }
  if ( (_DWORD)a2 != 96 )
  {
    if ( (unsigned int)a2 <= 0x60 )
    {
      if ( (_DWORD)a2 == 56 )
        return sub_3400BD0(a1, 0, a3, (unsigned int)v32, v33, 0, a7, 0);
      if ( (_DWORD)a2 == 58 )
        return sub_3400BD0(a1, 1, a3, (unsigned int)v32, a5, 0, a7, 0);
    }
    else if ( (_DWORD)a2 == 98 )
    {
      return (unsigned __int8 *)sub_33FE730(a1, a3, (unsigned int)v32, a5, 0, (__m128i)0x3FF0000000000000uLL);
    }
    return 0;
  }
  v24 = 0;
  if ( (a6 & 0x80) == 0 )
    v24 = (__m128i)0x8000000000000000LL;
  return (unsigned __int8 *)sub_33FE730(a1, a3, (unsigned int)v32, v33, 0, v24);
}
