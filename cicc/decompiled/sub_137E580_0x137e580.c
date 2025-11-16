// Function: sub_137E580
// Address: 0x137e580
//
__int64 __fastcall sub_137E580(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  __int64 v7; // rax
  unsigned int v8; // r13d
  __int64 v9; // rsi
  int v12; // eax
  int v13; // esi
  __int64 v14; // rcx
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // r13d
  __int64 v22; // rax
  __int64 v23; // r9
  __int64 v24; // rax
  unsigned int v25; // r13d
  __int64 *v26; // r14
  int v27; // eax
  __int64 v28; // rax
  int v29; // eax
  int v30; // r11d
  int v31; // [rsp+8h] [rbp-168h]
  __int64 v32; // [rsp+10h] [rbp-160h]
  int v33; // [rsp+1Ch] [rbp-154h]
  __int64 v35; // [rsp+28h] [rbp-148h]
  __int64 v36; // [rsp+28h] [rbp-148h]
  _QWORD *v37; // [rsp+30h] [rbp-140h] BYREF
  __int64 v38; // [rsp+38h] [rbp-138h]
  _QWORD v39[38]; // [rsp+40h] [rbp-130h] BYREF

  v6 = *(_QWORD *)(a1 + 40);
  v37 = v39;
  v38 = 0x2000000000LL;
  if ( v6 != *(_QWORD *)(a2 + 40) )
  {
    v39[0] = v6;
    LODWORD(v38) = 1;
    goto LABEL_3;
  }
  if ( a4 )
  {
    v12 = *(_DWORD *)(a4 + 24);
    if ( v12 )
    {
      v13 = v12 - 1;
      v14 = *(_QWORD *)(a4 + 8);
      v15 = (v12 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == v6 )
      {
LABEL_13:
        v8 = 1;
        if ( v16[1] )
          return v8;
      }
      else
      {
        v29 = 1;
        while ( v17 != -8 )
        {
          v30 = v29 + 1;
          v15 = v13 & (v29 + v15);
          v16 = (__int64 *)(v14 + 16LL * v15);
          v17 = *v16;
          if ( v6 == *v16 )
            goto LABEL_13;
          v29 = v30;
        }
      }
    }
  }
  v18 = a1 + 24;
  if ( a1 + 24 == v6 + 40 )
  {
LABEL_21:
    v19 = *(_QWORD *)(*(_QWORD *)(v6 + 56) + 80LL);
    if ( !v19 || (v8 = 0, v6 != v19 - 24) )
    {
      v35 = a1;
      v8 = 0;
      v20 = sub_157EBA0(v6);
      if ( v20 )
      {
        v21 = sub_15F4D60(v20);
        v33 = v21;
        v22 = sub_157EBA0(v6);
        v23 = v35;
        v32 = v22;
        v24 = (unsigned int)v38;
        v31 = v21;
        if ( v21 > HIDWORD(v38) - (unsigned __int64)(unsigned int)v38 )
        {
          sub_16CD150(&v37, v39, v21 + (unsigned __int64)(unsigned int)v38, 8);
          v24 = (unsigned int)v38;
          v23 = v35;
        }
        v25 = 0;
        v26 = &v37[v24];
        v27 = v31 + v24;
        if ( v33 )
        {
          do
          {
            v36 = v23;
            v28 = sub_15F4DF0(v32, v25);
            v23 = v36;
            if ( v26 )
              *v26 = v28;
            ++v26;
            ++v25;
          }
          while ( v33 != v25 );
          v27 = v38 + v31;
        }
        LODWORD(v38) = v27;
        if ( !v27 )
          goto LABEL_20;
        v6 = *(_QWORD *)(v23 + 40);
LABEL_3:
        v7 = *(_QWORD *)(*(_QWORD *)(v6 + 56) + 80LL);
        if ( v7 )
        {
          v7 -= 24;
          v8 = 1;
          if ( v7 == v6 )
          {
LABEL_7:
            if ( v37 != v39 )
              _libc_free((unsigned __int64)v37);
            return v8;
          }
        }
        v9 = *(_QWORD *)(a2 + 40);
        if ( v9 != v7 )
        {
          v8 = sub_137E120((__int64)&v37, v9, a3, a4);
          goto LABEL_7;
        }
LABEL_20:
        v8 = 0;
        goto LABEL_7;
      }
    }
  }
  else
  {
    while ( !v18 || a2 != v18 - 24 )
    {
      v18 = *(_QWORD *)(v18 + 8);
      if ( v18 == v6 + 40 )
        goto LABEL_21;
    }
    return 1;
  }
  return v8;
}
