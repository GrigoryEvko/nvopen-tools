// Function: sub_16D2BE0
// Address: 0x16d2be0
//
__int64 __fastcall sub_16D2BE0(const __m128i *a1, unsigned int a2, unsigned __int64 *a3)
{
  unsigned int v4; // ebx
  __int64 v5; // rax
  unsigned int v6; // r12d
  _BYTE *v7; // rdx
  unsigned int v8; // r12d
  int v9; // r13d
  unsigned int v10; // edx
  unsigned int v11; // r15d
  char *v12; // rdx
  char v13; // al
  unsigned int v14; // eax
  __int64 v15; // r15
  unsigned int v17; // esi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rcx
  unsigned int v20; // eax
  unsigned int v21; // eax
  unsigned int v22; // eax
  unsigned int v23; // [rsp+8h] [rbp-78h]
  unsigned __int128 v24; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v25; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-58h]
  _QWORD *v27; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-48h]
  _QWORD *v29; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+48h] [rbp-38h]

  v4 = a2;
  v24 = (unsigned __int128)_mm_loadu_si128(a1);
  if ( !a2 )
    v4 = sub_16D1EB0((__int64)&v24);
  v5 = *((_QWORD *)&v24 + 1);
  v6 = 1;
  if ( *((_QWORD *)&v24 + 1) )
  {
    v7 = (_BYTE *)v24;
    while ( *v7 == 48 )
    {
      --v5;
      *(_QWORD *)&v24 = ++v7;
      *((_QWORD *)&v24 + 1) = v5;
      if ( !v5 )
      {
        if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
          j_j___libc_free_0_0(*a3);
        *a3 = 0;
        v6 = 0;
        *((_DWORD *)a3 + 2) = 64;
        return v6;
      }
    }
    if ( v4 <= 1 )
    {
      v11 = *((_DWORD *)a3 + 2);
      if ( !v11 )
      {
        v10 = 0;
        v8 = 0;
        v9 = 1;
LABEL_12:
        v11 = v10;
        goto LABEL_13;
      }
      v8 = 0;
      v9 = 1;
    }
    else
    {
      v8 = 0;
      do
        v9 = 1 << ++v8;
      while ( 1 << v8 < v4 );
      v10 = v8 * v5;
      v11 = *((_DWORD *)a3 + 2);
      if ( v11 <= v8 * (unsigned int)v5 )
      {
        if ( v11 < v10 )
        {
          v23 = v8 * v5;
          sub_16A5C50((__int64)&v29, (const void **)a3, v10);
          v10 = v23;
          if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
          {
            j_j___libc_free_0_0(*a3);
            v10 = v23;
          }
          *a3 = (unsigned __int64)v29;
          *((_DWORD *)a3 + 2) = v30;
        }
        goto LABEL_12;
      }
    }
LABEL_13:
    v26 = 1;
    v25 = 0;
    v28 = 1;
    v27 = 0;
    if ( v4 != v9 )
    {
      v30 = v11;
      if ( v11 > 0x40 )
      {
        sub_16A4EF0((__int64)&v29, v4, 0);
        if ( v26 > 0x40 && v25 )
        {
          j_j___libc_free_0_0(v25);
          v25 = (unsigned __int64)v29;
          v21 = v30;
          v30 = v11;
          v26 = v21;
        }
        else
        {
          v25 = (unsigned __int64)v29;
          v22 = v30;
          v30 = v11;
          v26 = v22;
        }
        sub_16A4EF0((__int64)&v29, 0, 0);
        if ( v28 > 0x40 && v27 )
          j_j___libc_free_0_0(v27);
      }
      else
      {
        v26 = v11;
        v29 = 0;
        v25 = v4 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v11);
      }
      v27 = v29;
      v28 = v30;
    }
    if ( *((_DWORD *)a3 + 2) > 0x40u )
    {
      *(_QWORD *)*a3 = 0;
      memset((void *)(*a3 + 8), 0, 8 * (unsigned int)(((unsigned __int64)*((unsigned int *)a3 + 2) + 63) >> 6) - 8);
    }
    else
    {
      *a3 = 0;
    }
    v12 = (char *)v24;
    if ( *((_QWORD *)&v24 + 1) )
    {
      v13 = *(_BYTE *)v24;
      if ( *(char *)v24 > 47 )
        goto LABEL_31;
LABEL_21:
      if ( (unsigned __int8)(v13 - 65) <= 0x19u )
      {
        v14 = (char)(v13 - 55);
        while ( v4 > v14 )
        {
          v15 = v14;
          if ( v4 == v9 )
          {
            v17 = *((_DWORD *)a3 + 2);
            if ( v17 <= 0x40 )
            {
              v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
              v19 = 0;
              if ( v17 != v8 )
                v19 = v18 & (*a3 << v8);
              *a3 = v19;
              goto LABEL_50;
            }
            sub_16A7DC0((__int64 *)a3, v8);
            v20 = *((_DWORD *)a3 + 2);
            if ( v20 <= 0x40 )
            {
              v12 = (char *)v24;
              v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v20;
LABEL_50:
              *a3 = (v15 | *a3) & v18;
              goto LABEL_28;
            }
            *(_QWORD *)*a3 |= v15;
            v12 = (char *)v24;
          }
          else
          {
            sub_16A7C10((__int64)a3, (__int64 *)&v25);
            if ( v28 > 0x40 )
            {
              *v27 = v15;
              memset(v27 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v28 + 63) >> 6) - 8);
            }
            else
            {
              v27 = (_QWORD *)(v15 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v28));
            }
            sub_16A7200((__int64)a3, (__int64 *)&v27);
            v12 = (char *)v24;
          }
LABEL_28:
          if ( !*((_QWORD *)&v24 + 1) )
          {
            *(_QWORD *)&v24 = v12;
            goto LABEL_45;
          }
          v24 = __PAIR128__(*((unsigned __int64 *)&v24 + 1), (unsigned __int64)++v12) + __PAIR128__(-1, 0);
          if ( !*((_QWORD *)&v24 + 1) )
            goto LABEL_45;
          v13 = *v12;
          if ( *v12 <= 47 )
            goto LABEL_21;
LABEL_31:
          if ( v13 > 57 )
          {
            if ( v13 <= 96 )
              goto LABEL_21;
            if ( v13 <= 122 )
            {
              v14 = (char)(v13 - 87);
              continue;
            }
            break;
          }
          v14 = (char)(v13 - 48);
        }
      }
      v6 = 1;
    }
    else
    {
LABEL_45:
      v6 = 0;
    }
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
  }
  return v6;
}
