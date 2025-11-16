// Function: sub_2D940D0
// Address: 0x2d940d0
//
__int64 __fastcall sub_2D940D0(__int64 a1, void *a2, size_t a3, unsigned int a4, unsigned __int64 *a5)
{
  unsigned __int64 *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // r8
  __int64 (__fastcall *v12)(__int64, unsigned __int64 *, __int64 *, __int64, __int64, __int64, void **, __int64, __int64, _QWORD, _QWORD); // rax
  __int64 v13; // rax
  unsigned __int64 *v14; // rbx
  unsigned __int64 *v15; // r14
  volatile signed __int32 *v16; // rbx
  signed __int32 v17; // eax
  unsigned int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // r14
  unsigned int v22; // ebx
  __int64 v23; // rax
  unsigned int v24; // ebx
  __int64 v25; // rdx
  __int64 v26; // r14
  unsigned __int64 v27; // rax
  signed __int32 v28; // eax
  __int64 v32; // [rsp+28h] [rbp-298h]
  __int64 v33; // [rsp+28h] [rbp-298h]
  __int64 v34; // [rsp+30h] [rbp-290h]
  __int64 v35; // [rsp+38h] [rbp-288h]
  __int64 v36; // [rsp+48h] [rbp-278h] BYREF
  __m128i *v37[2]; // [rsp+50h] [rbp-270h] BYREF
  _BYTE v38[16]; // [rsp+60h] [rbp-260h] BYREF
  __int64 *v39; // [rsp+70h] [rbp-250h] BYREF
  __int64 v40; // [rsp+78h] [rbp-248h]
  __int64 v41; // [rsp+80h] [rbp-240h] BYREF
  __int64 v42[2]; // [rsp+90h] [rbp-230h] BYREF
  _QWORD v43[2]; // [rsp+A0h] [rbp-220h] BYREF
  unsigned __int64 v44[2]; // [rsp+B0h] [rbp-210h] BYREF
  __int64 v45; // [rsp+C0h] [rbp-200h] BYREF
  void *s1; // [rsp+F0h] [rbp-1D0h] BYREF
  size_t n; // [rsp+F8h] [rbp-1C8h]
  _QWORD v48[2]; // [rsp+100h] [rbp-1C0h] BYREF
  __int16 v49; // [rsp+110h] [rbp-1B0h]
  volatile signed __int32 *v50; // [rsp+118h] [rbp-1A8h]
  __int64 *v51; // [rsp+128h] [rbp-198h]
  __int64 v52; // [rsp+138h] [rbp-188h] BYREF
  __int64 *v53; // [rsp+188h] [rbp-138h]
  __int64 v54; // [rsp+198h] [rbp-128h] BYREF
  __int64 *v55; // [rsp+1A8h] [rbp-118h]
  __int64 v56; // [rsp+1B8h] [rbp-108h] BYREF
  __int64 *v57; // [rsp+1C8h] [rbp-F8h]
  __int64 v58; // [rsp+1D8h] [rbp-E8h] BYREF
  __int64 *v59; // [rsp+1E8h] [rbp-D8h]
  __int64 v60; // [rsp+1F8h] [rbp-C8h] BYREF
  __int64 *v61; // [rsp+208h] [rbp-B8h]
  __int64 v62; // [rsp+218h] [rbp-A8h] BYREF
  __int64 *v63; // [rsp+228h] [rbp-98h]
  __int64 v64; // [rsp+238h] [rbp-88h] BYREF
  unsigned __int64 *v65; // [rsp+248h] [rbp-78h]
  unsigned __int64 *v66; // [rsp+250h] [rbp-70h]
  __int64 v67; // [rsp+258h] [rbp-68h]
  __int64 *v68; // [rsp+268h] [rbp-58h]
  __int64 v69; // [rsp+278h] [rbp-48h] BYREF

  s1 = a2;
  v49 = 261;
  n = a3;
  sub_CC9F70((__int64)v44, &s1);
  v37[1] = 0;
  v37[0] = (__m128i *)v38;
  v38[0] = 0;
  sub_2D90EA0((__int64 *)&s1);
  v32 = sub_C0DB40(s1, n, v44, v37);
  if ( s1 != v48 )
    j_j___libc_free_0((unsigned __int64)s1);
  if ( v32 )
  {
    sub_2D92D10((__int64 *)&v39);
    if ( !v40 )
      sub_2240AE0((unsigned __int64 *)&v39, a5);
    v34 = sub_2D91230();
    v6 = v44;
    v35 = sub_2D91110();
    sub_2D91E40((__int64)&s1, v44);
    sub_2D92DA0((__int64)v42, (__int64)v44, v7, v8, v9, v10);
    v11 = (_QWORD *)v42[0];
    v36 = v34;
    v12 = *(__int64 (__fastcall **)(__int64, unsigned __int64 *, __int64 *, __int64, __int64, __int64, void **, __int64, __int64, _QWORD, _QWORD))(v32 + 96);
    if ( v12 )
    {
      v6 = v44;
      v13 = v12(v32, v44, v39, v40, v42[0], v42[1], &s1, v35, v34, a4, 0);
      v11 = (_QWORD *)v42[0];
      v33 = v13;
    }
    else
    {
      v33 = 0;
    }
    if ( v11 != v43 )
    {
      v6 = (unsigned __int64 *)(v43[0] + 1LL);
      j_j___libc_free_0((unsigned __int64)v11);
    }
    if ( v68 != &v69 )
    {
      v6 = (unsigned __int64 *)(v69 + 1);
      j_j___libc_free_0((unsigned __int64)v68);
    }
    v14 = v66;
    v15 = v65;
    if ( v66 != v65 )
    {
      do
      {
        if ( (unsigned __int64 *)*v15 != v15 + 2 )
        {
          v6 = (unsigned __int64 *)(v15[2] + 1);
          j_j___libc_free_0(*v15);
        }
        v15 += 4;
      }
      while ( v14 != v15 );
      v15 = v65;
    }
    if ( v15 )
    {
      v6 = (unsigned __int64 *)(v67 - (_QWORD)v15);
      j_j___libc_free_0((unsigned __int64)v15);
    }
    if ( v63 != &v64 )
    {
      v6 = (unsigned __int64 *)(v64 + 1);
      j_j___libc_free_0((unsigned __int64)v63);
    }
    if ( v61 != &v62 )
    {
      v6 = (unsigned __int64 *)(v62 + 1);
      j_j___libc_free_0((unsigned __int64)v61);
    }
    if ( v59 != &v60 )
    {
      v6 = (unsigned __int64 *)(v60 + 1);
      j_j___libc_free_0((unsigned __int64)v59);
    }
    if ( v57 != &v58 )
    {
      v6 = (unsigned __int64 *)(v58 + 1);
      j_j___libc_free_0((unsigned __int64)v57);
    }
    if ( v55 != &v56 )
    {
      v6 = (unsigned __int64 *)(v56 + 1);
      j_j___libc_free_0((unsigned __int64)v55);
    }
    if ( v53 != &v54 )
    {
      v6 = (unsigned __int64 *)(v54 + 1);
      j_j___libc_free_0((unsigned __int64)v53);
    }
    if ( v51 != &v52 )
    {
      v6 = (unsigned __int64 *)(v52 + 1);
      j_j___libc_free_0((unsigned __int64)v51);
    }
    v16 = v50;
    if ( v50 )
    {
      if ( &_pthread_key_create )
      {
        v17 = _InterlockedExchangeAdd(v50 + 2, 0xFFFFFFFF);
      }
      else
      {
        v17 = *((_DWORD *)v50 + 2);
        *((_DWORD *)v50 + 2) = v17 - 1;
      }
      if ( v17 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *, unsigned __int64 *))(*(_QWORD *)v16 + 16LL))(v16, v6);
        if ( &_pthread_key_create )
        {
          v28 = _InterlockedExchangeAdd(v16 + 3, 0xFFFFFFFF);
        }
        else
        {
          v28 = *((_DWORD *)v16 + 3);
          *((_DWORD *)v16 + 3) = v28 - 1;
        }
        if ( v28 == 1 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 24LL))(v16);
      }
    }
    if ( v33 )
    {
      *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
      *(_QWORD *)a1 = v33;
    }
    else
    {
      s1 = "could not allocate target machine for ";
      v48[0] = a2;
      v48[1] = a3;
      v49 = 1283;
      v19 = sub_C63BB0();
      v21 = v20;
      v22 = v19;
      sub_CA0F50(v42, &s1);
      sub_C63F00(&v36, (__int64)v42, v22, v21);
      if ( (_QWORD *)v42[0] != v43 )
        j_j___libc_free_0(v42[0]);
      v23 = v36;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v23 & 0xFFFFFFFFFFFFFFFELL;
    }
    if ( v39 != &v41 )
      j_j___libc_free_0((unsigned __int64)v39);
  }
  else
  {
    s1 = v37;
    v49 = 260;
    v24 = sub_C63BB0();
    v26 = v25;
    sub_CA0F50(v42, &s1);
    sub_C63F00((__int64 *)&v39, (__int64)v42, v24, v26);
    if ( (_QWORD *)v42[0] != v43 )
      j_j___libc_free_0(v42[0]);
    v27 = (unsigned __int64)v39;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v27 & 0xFFFFFFFFFFFFFFFELL;
  }
  if ( v37[0] != (__m128i *)v38 )
    j_j___libc_free_0((unsigned __int64)v37[0]);
  if ( (__int64 *)v44[0] != &v45 )
    j_j___libc_free_0(v44[0]);
  return a1;
}
