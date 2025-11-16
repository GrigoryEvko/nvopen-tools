// Function: sub_1319320
// Address: 0x1319320
//
_BYTE *__fastcall sub_1319320(__int64 a1, __int64 a2)
{
  unsigned int i; // ebx
  __int64 v3; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx

  *(_DWORD *)a2 = 0;
  *(_DWORD *)(a2 + 4) = 0;
  if ( a2 == *(_QWORD *)(a1 + 144) )
  {
    sub_13177F0(a2, 0);
    if ( a2 != *(_QWORD *)(a1 + 136) )
      goto LABEL_3;
LABEL_14:
    sub_13177F0(a2, 1u);
    goto LABEL_3;
  }
  if ( a2 == *(_QWORD *)(a1 + 136) )
    goto LABEL_14;
LABEL_3:
  *(_QWORD *)(a2 + 10392) = 0;
  *(_QWORD *)(a2 + 10400) = 0;
  if ( *(_BYTE *)a1 && *(_QWORD *)(a1 + 296) == a2 )
  {
    v5 = a1 + 256;
    v6 = *(_QWORD *)(a1 + 424);
    *(_QWORD *)(a1 + 256) = a1 + 256;
    *(_QWORD *)(a1 + 264) = a1 + 256;
    v7 = *(_QWORD *)(a2 + 10392);
    if ( v7 )
    {
      *(_QWORD *)(a1 + 256) = *(_QWORD *)(v7 + 8);
      *(_QWORD *)(*(_QWORD *)(a2 + 10392) + 8LL) = v5;
      *(_QWORD *)(a1 + 264) = **(_QWORD **)(a1 + 264);
      **(_QWORD **)(*(_QWORD *)(a2 + 10392) + 8LL) = *(_QWORD *)(a2 + 10392);
      **(_QWORD **)(a1 + 264) = v5;
      v5 = *(_QWORD *)(a1 + 256);
    }
    *(_QWORD *)(a2 + 10392) = v5;
    v8 = a1 + 272;
    *(_QWORD *)(a1 + 272) = a1 + 272;
    *(_QWORD *)(a1 + 280) = a1 + 272;
    *(_QWORD *)(a1 + 288) = v6 + 8;
    v9 = *(_QWORD *)(a2 + 10400);
    if ( v9 )
    {
      *(_QWORD *)(a1 + 272) = *(_QWORD *)(v9 + 8);
      *(_QWORD *)(*(_QWORD *)(a2 + 10400) + 8LL) = v8;
      *(_QWORD *)(a1 + 280) = **(_QWORD **)(a1 + 280);
      **(_QWORD **)(*(_QWORD *)(a2 + 10400) + 8LL) = *(_QWORD *)(a2 + 10400);
      **(_QWORD **)(a1 + 280) = v8;
      v8 = *(_QWORD *)(a1 + 272);
    }
    *(_QWORD *)(a2 + 10400) = v8;
  }
  for ( i = 0; dword_4F96B60 > i; ++i )
  {
    v3 = i;
    sub_131C800(a1, (__int64 *)(a2 + 224 * v3 + 78984));
  }
  sub_130B060(a1, (__int64 *)(a2 + 10536));
  sub_131C570(a1, *(_QWORD *)(a2 + 78936));
  sub_134AD70(a1, a2 + 10648);
  return sub_130B060(a1, (__int64 *)(a2 + 10408));
}
