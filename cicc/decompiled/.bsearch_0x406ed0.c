// Function: .bsearch
// Address: 0x406ed0
//
// attributes: thunk
void *bsearch(const void *key, const void *base, size_t nmemb, size_t size, __compar_fn_t compar)
{
  return bsearch(key, base, nmemb, size, compar);
}
