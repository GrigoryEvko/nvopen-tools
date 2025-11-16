// Function: sub_19E6D50
// Address: 0x19e6d50
//
__int64 __fastcall sub_19E6D50(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 *v5; // rbx
  __int64 *v6; // rax
  int v7; // r11d
  __int64 v8; // r15
  __int64 v9; // r14
  unsigned int v10; // r9d
  int v11; // r13d
  __int64 v12; // r8
  unsigned int v13; // esi
  unsigned int v14; // edi
  __int64 *v15; // rsi
  __int64 v16; // r10
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 *v27; // r15
  unsigned int v28; // ebx
  __int64 *v29; // r13
  __int64 v30; // r14
  char v31; // r10
  unsigned int v32; // eax
  int v33; // esi
  __int64 v34; // [rsp+10h] [rbp-100h]
  __int64 *v35; // [rsp+18h] [rbp-F8h]
  int v36; // [rsp+18h] [rbp-F8h]
  __int64 *v37; // [rsp+20h] [rbp-F0h] BYREF
  _QWORD v38[4]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v39[4]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v40; // [rsp+80h] [rbp-90h] BYREF
  __int64 *v41; // [rsp+88h] [rbp-88h]
  __int64 *v42; // [rsp+A0h] [rbp-70h] BYREF
  __int64 *v43; // [rsp+A8h] [rbp-68h]
  __int64 v44; // [rsp+B0h] [rbp-60h]
  __int64 v45; // [rsp+B8h] [rbp-58h]
  __int64 *v46; // [rsp+C0h] [rbp-50h] BYREF
  __int64 *v47; // [rsp+C8h] [rbp-48h]
  __int64 v48; // [rsp+D0h] [rbp-40h]
  __int64 v49; // [rsp+D8h] [rbp-38h]

  if ( *(_DWORD *)(a2 + 184) )
  {
    v4 = *(_QWORD *)(a2 + 16);
    if ( !v4 || *(_BYTE *)(v4 + 16) != 55 )
    {
      sub_19E54A0(&v37, (__int64 *)(a2 + 56));
      sub_19E54A0(v38, (__int64 *)(a2 + 56));
      v5 = (__int64 *)v38[0];
      if ( (__int64 *)v38[0] != v37 )
      {
        do
        {
          if ( *(_BYTE *)(*v5 + 16) == 55 )
            break;
          do
            ++v5;
          while ( (__int64 *)v38[1] != v5 && (unsigned __int64)*v5 >= 0xFFFFFFFFFFFFFFFELL );
        }
        while ( v37 != v5 );
      }
      sub_19E54A0(v39, (__int64 *)(a2 + 56));
      sub_19E68D0(&v40, (__int64 *)(a2 + 56), *(_QWORD **)(a2 + 72));
      v6 = (__int64 *)v40;
      if ( v40 != v39[0] )
      {
        do
        {
          if ( *(_BYTE *)(*v6 + 16) == 55 )
            break;
          do
            ++v6;
          while ( v41 != v6 && (unsigned __int64)*v6 >= 0xFFFFFFFFFFFFFFFELL );
        }
        while ( (__int64 *)v39[0] != v6 );
      }
      if ( v5 == v6 )
      {
        v8 = 0;
      }
      else
      {
        v7 = *(_DWORD *)(a1 + 2416);
        v8 = 0;
        v9 = *(_QWORD *)(a1 + 2400);
        v10 = -1;
        v11 = v7 - 1;
        do
        {
          v12 = *v6;
          v13 = 0;
          if ( v7 )
          {
            v14 = v11 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
            v15 = (__int64 *)(v9 + 16LL * v14);
            v16 = *v15;
            if ( v12 == *v15 )
            {
LABEL_20:
              v13 = *((_DWORD *)v15 + 2);
            }
            else
            {
              v33 = 1;
              while ( v16 != -8 )
              {
                v14 = v11 & (v33 + v14);
                v36 = v33 + 1;
                v15 = (__int64 *)(v9 + 16LL * v14);
                v16 = *v15;
                if ( v12 == *v15 )
                  goto LABEL_20;
                v33 = v36;
              }
              v13 = 0;
            }
          }
          if ( v13 < v10 )
          {
            v10 = v13;
            v8 = *v6;
          }
          do
            ++v6;
          while ( v41 != v6 && (unsigned __int64)(*v6 + 2) <= 1 );
          while ( (__int64 *)v39[0] != v6 )
          {
LABEL_26:
            if ( *(_BYTE *)(*v6 + 16) == 55 )
              break;
            while ( v41 != ++v6 )
            {
              if ( (unsigned __int64)*v6 < 0xFFFFFFFFFFFFFFFELL )
              {
                if ( (__int64 *)v39[0] != v6 )
                  goto LABEL_26;
                goto LABEL_30;
              }
            }
          }
LABEL_30:
          ;
        }
        while ( v5 != v6 );
      }
      v4 = v8;
    }
    return sub_19E6CE0(a1, v4);
  }
  else
  {
    v18 = *(unsigned int *)(a2 + 156);
    v19 = *(_QWORD *)(a2 + 144);
    v20 = *(_QWORD *)(a2 + 136);
    if ( *(_DWORD *)(a2 + 156) - *(_DWORD *)(a2 + 160) == 1 )
    {
      if ( v20 != v19 )
        v18 = *(unsigned int *)(a2 + 152);
      v46 = *(__int64 **)(a2 + 144);
      v47 = (__int64 *)(v19 + 8 * v18);
      sub_19E4730((__int64)&v46);
      v48 = a2 + 128;
      v49 = *(_QWORD *)(a2 + 128);
      return *v46;
    }
    else
    {
      v21 = (__int64 *)(v20 + 8 * v18);
      if ( v20 != v19 )
        v21 = (__int64 *)(v19 + 8LL * *(unsigned int *)(a2 + 152));
      v46 = v21;
      v47 = v21;
      sub_19E4730((__int64)&v46);
      v22 = *(_QWORD *)(a2 + 128);
      v48 = a2 + 128;
      v49 = v22;
      v23 = *(_QWORD *)(a2 + 144);
      if ( v23 == *(_QWORD *)(a2 + 136) )
        v24 = (__int64 *)(v23 + 8LL * *(unsigned int *)(a2 + 156));
      else
        v24 = (__int64 *)(v23 + 8LL * *(unsigned int *)(a2 + 152));
      v42 = *(__int64 **)(a2 + 144);
      v25 = a1 + 2392;
      v43 = v24;
      sub_19E4730((__int64)&v42);
      v26 = *(_QWORD *)(a2 + 128);
      v44 = a2 + 128;
      v27 = v43;
      v28 = -1;
      v45 = v26;
      v29 = v42;
      v34 = 0;
      v35 = v46;
      while ( v35 != v29 )
      {
        v30 = *v29;
        if ( (unsigned int)*(unsigned __int8 *)(*v29 + 16) - 21 > 1 )
        {
          v32 = sub_19E5210(v25, *v29);
        }
        else
        {
          v39[0] = *(_QWORD *)(v30 + 72);
          v31 = sub_154CC80(v25, v39, &v40);
          v32 = 0;
          if ( v31 )
            v32 = *(_DWORD *)(v40 + 8);
        }
        if ( v32 < v28 )
        {
          v34 = v30;
          v28 = v32;
        }
        for ( ++v29; v27 != v29; ++v29 )
        {
          if ( (unsigned __int64)*v29 < 0xFFFFFFFFFFFFFFFELL )
            break;
        }
      }
    }
    return v34;
  }
}
